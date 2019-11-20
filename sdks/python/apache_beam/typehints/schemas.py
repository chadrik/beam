#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Support for mapping python types to proto Schemas and back again.

Python              Schema
np.int8     <-----> BYTE
np.int16    <-----> INT16
np.int32    <-----> INT32
np.int64    <-----> INT64
int         ---/
np.float32  <-----> FLOAT
np.float64  <-----> DOUBLE
float       ---/
bool        <-----> BOOLEAN

The mappings for STRING and BYTES are different between python 2 and python 3,
because of the changes to str:
py3:
str/unicode <-----> STRING
bytes       <-----> BYTES
ByteString  ---/

py2:
str will be rejected since it is ambiguous.
unicode     <-----> STRING
ByteString  <-----> BYTES
"""

from __future__ import absolute_import

import sys
from collections import OrderedDict
from typing import Any
from typing import ByteString
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from uuid import uuid4

import numpy as np
from past.builtins import unicode

from apache_beam.portability.api import schema_pb2
from apache_beam.typehints.native_type_compatibility import _get_args
from apache_beam.typehints.native_type_compatibility import _match_is_exactly_mapping
from apache_beam.typehints.native_type_compatibility import _match_is_named_tuple
from apache_beam.typehints.native_type_compatibility import _match_is_optional
from apache_beam.typehints.native_type_compatibility import _safe_issubclass
from apache_beam.typehints.native_type_compatibility import extract_optional_type

SchemaConverterT = TypeVar('SchemaConverterT', bound='SchemaConverter')


# Registry of typings for a schema by UUID
class SchemaTypeRegistry(object):
  def __init__(self):
    self.by_id = {}  # type: Dict[SchemaConverter]

  def add(self, converter):
    # type: (SchemaConverter) -> None
    from apache_beam import coders
    typing = converter.type
    if hasattr(typing, '__beam_schema_id__'):
      if typing.__beam_schema_id__ != converter.schema.id:
        raise ValueError("Type '%s' was previously registered with a "
                         "different id" % typing)
    else:
      setattr(typing, '__beam_schema_id__', converter.schema.id)
      self.by_id[converter.schema.id] = converter
      coders.registry.register_coder(converter.type, coders.RowCoder)

  def get_by_schema(self, schema):
    # type: (schema_pb2.Schema) -> Optional[SchemaConverter]
    return self.by_id.get(schema.id, None)

  def get_by_type(self, type_):
    # type: (Type) -> Optional[SchemaConverter]
    if hasattr(type_, '__beam_schema_id__'):
      result = SCHEMA_REGISTRY.by_id.get(type_.__beam_schema_id__, None)
      if result is None:
        raise ValueError("Type '%s' was previously registered but its schema "
                         "is missing from the registry" % type_)
      return result
    return None

SCHEMA_REGISTRY = SchemaTypeRegistry()


# Bi-directional mappings
_PRIMITIVES = (
    (np.int8, schema_pb2.BYTE),
    (np.int16, schema_pb2.INT16),
    (np.int32, schema_pb2.INT32),
    (np.int64, schema_pb2.INT64),
    (np.float32, schema_pb2.FLOAT),
    (np.float64, schema_pb2.DOUBLE),
    (unicode, schema_pb2.STRING),
    (bool, schema_pb2.BOOLEAN),
    (bytes if sys.version_info.major >= 3 else ByteString,
     schema_pb2.BYTES),
)  # type: Tuple[Tuple[Type, int], ...]

PRIMITIVE_TO_ATOMIC_TYPE = dict((typ, atomic) for typ, atomic in _PRIMITIVES)
ATOMIC_TYPE_TO_PRIMITIVE = dict((atomic, typ) for typ, atomic in _PRIMITIVES)

# One-way mappings
PRIMITIVE_TO_ATOMIC_TYPE.update({
    # In python 2, this is a no-op because we define it as the bi-directional
    # mapping above. This just ensures the one-way mapping is defined in python
    # 3.
    ByteString: schema_pb2.BYTES,
    # Allow users to specify a native int, and use INT64 as the cross-language
    # representation. Technically ints have unlimited precision, but RowCoder
    # should throw an error if it sees one with a bit width > 64 when encoding.
    int: schema_pb2.INT64,
    float: schema_pb2.DOUBLE,
})


def typing_to_runner_api(type_):
  # type: (Type) -> schema_pb2.FieldType
  converter = SCHEMA_REGISTRY.get_by_type(type_)
  if converter is None:
    converter = SchemaConverter.from_type(type_)
    if converter is not None:
      SCHEMA_REGISTRY.add(converter)
  if converter is not None:
    return schema_pb2.FieldType(
        row_type=schema_pb2.RowType(
            schema=converter.schema))

  # All concrete types (other than NamedTuple sub-classes) should map to
  # a supported primitive type.
  elif type_ in PRIMITIVE_TO_ATOMIC_TYPE:
    return schema_pb2.FieldType(atomic_type=PRIMITIVE_TO_ATOMIC_TYPE[type_])

  elif sys.version_info.major == 2 and type_ == str:
    raise ValueError(
        "type 'str' is not supported in python 2. Please use 'unicode' or "
        "'typing.ByteString' instead to unambiguously indicate if this is a "
        "UTF-8 string or a byte array."
    )

  elif _match_is_exactly_mapping(type_):
    key_type, value_type = map(typing_to_runner_api, _get_args(type_))
    return schema_pb2.FieldType(
        map_type=schema_pb2.MapType(key_type=key_type, value_type=value_type))

  elif _match_is_optional(type_):
    # It's possible that a user passes us Optional[Optional[T]], but in python
    # typing this is indistinguishable from Optional[T] - both resolve to
    # Union[T, None] - so there's no need to check for that case here.
    result = typing_to_runner_api(extract_optional_type(type_))
    result.nullable = True
    return result

  elif _safe_issubclass(type_, Sequence):
    element_type = typing_to_runner_api(_get_args(type_)[0])
    return schema_pb2.FieldType(
        array_type=schema_pb2.ArrayType(element_type=element_type))

  raise ValueError("Unsupported type: %s" % type_)


def typing_from_runner_api(fieldtype_proto):
  # type: (schema_pb2.FieldType) -> Type
  if fieldtype_proto.nullable:
    # In order to determine the inner type, create a copy of fieldtype_proto
    # with nullable=False and pass back to typing_from_runner_api
    base_type = schema_pb2.FieldType()
    base_type.CopyFrom(fieldtype_proto)
    base_type.nullable = False
    return Optional[typing_from_runner_api(base_type)]

  type_info = fieldtype_proto.WhichOneof("type_info")
  if type_info == "atomic_type":
    try:
      return ATOMIC_TYPE_TO_PRIMITIVE[fieldtype_proto.atomic_type]
    except KeyError:
      raise ValueError("Unsupported atomic type: {0}".format(
          fieldtype_proto.atomic_type))
  elif type_info == "array_type":
    return Sequence[typing_from_runner_api(
        fieldtype_proto.array_type.element_type)]
  elif type_info == "map_type":
    return Mapping[
        typing_from_runner_api(fieldtype_proto.map_type.key_type),
        typing_from_runner_api(fieldtype_proto.map_type.value_type)
    ]
  elif type_info == "row_type":
    return type_from_schema(fieldtype_proto.row_type.schema)

  elif type_info == "logical_type":
    pass  # TODO


class SchemaConverter(object):
  converters = []  # type: List[Type[SchemaConverter]]

  def __init__(self, type=None, schema=None):
    self._type = type
    self._schema = schema
    if type is None and schema is None:
      raise ValueError("'type' and 'schema' cannot both be None")

  @classmethod
  def register(cls, converter):
    # type: (Type[SchemaConverterT]) -> Type[SchemaConverterT]
    """Register a converter class
    """
    cls.converters.append(converter)
    return converter

  @classmethod
  def claim_type(cls, type_):
    # type: (Type) -> bool
    raise NotImplementedError

  @classmethod
  def from_type(cls, type_):
    # type: (Type) -> Optional[SchemaConverter]
    """Find and instantiate a converter for the given type.
    """
    for converter_cls in cls.converters:
      if converter_cls.claim_type(type_):
        return converter_cls(type=type_)

  @classmethod
  def from_schema(cls, schema):
    # type: (schema_pb2.Schema) -> Optional[SchemaConverter]
    """Instantiate this converter with the given schema.
    """
    return cls(schema=schema)

  def get_fields(self):
    # type: () -> List[schema_pb2.Field]
    """Return a list of fields for `self.type`

    Subclasses are expected to override this
    """
    raise NotImplementedError

  def get_field_values(self, instance):
    """Return a list of values for `instance`
    """
    raise NotImplementedError

  @property
  def schema(self):
    # type: () -> schema_pb2.Schema
    """Get the schema, creating it from the type if necessary.
    """
    if self._schema is None:
      type_id = str(uuid4())
      self._schema = schema_pb2.Schema(fields=self.get_fields(), id=type_id)
    return self._schema

  @property
  def type(self):
    # type: () -> Type
    """Get the type, creating it from the schema if necessary
    """
    if self._type is None:
      type_name = 'BeamSchema_{}'.format(self._schema.id.replace('-', '_'))
      self._type = self.make_type(type_name)
    return self._type

  def make_type(self, type_name):
    # type: (str) -> Type
    """Create the class for this schema.

    :param type_name: the suggested name for the class
    :return: type
    """
    raise NotImplementedError

  def get_constructor(self):
    # type: () -> Callable[[OrderedDict[str, Any]], Type]
    """Return a constructor for this class.

    :return: callable that takes a OrderedDict of field name to values and
      returns an instance of `self.type`.
    """
    raise NotImplementedError


@SchemaConverter.register
class NamedTupleSchemaConverter(SchemaConverter):
  @classmethod
  def claim_type(cls, type_):
    return _match_is_named_tuple(type_)

  def get_fields(self):
    return [
        schema_pb2.Field(
            name=name, type=typing_to_runner_api(self._type._field_types[name]))
        for name in self._type._fields]

  def get_field_values(self, instance):
    return [getattr(instance, f.name) for f in self.schema.fields]

  def make_type(self, type_name):
    return NamedTuple(type_name,
                      [(field.name, typing_from_runner_api(field.type))
                       for field in self._schema.fields])

  def get_constructor(self):
    def constructor(values):
      return self._type(*values.values())
    return constructor


if sys.version_info[:2] >= (3, 6):
  # __annotations__ is a dict, and order is not stable for dict in
  # python < 3.6. Without stable ordering, RowCoder will not work reliably.

  @SchemaConverter.register
  class TypedDictSchemaConverter(SchemaConverter):
    @classmethod
    def claim_type(cls, type_):
      return (isinstance(type_, type) and issubclass(type_, dict)
              and type_ is not dict and hasattr(type_, '__annotations__'))

    def get_fields(self):
      return [
          schema_pb2.Field(
              name=name, type=typing_to_runner_api(field_type))
          for name, field_type in self._type.__annotations__.items()]

    def get_field_values(self, instance):
      return [instance[f.name] for f in self.schema.fields]

    def make_type(self, type_name):
      import mypy_extensions
      return mypy_extensions.TypedDict(
          type_name,
          [(field.name, typing_from_runner_api(field.type))
           for field in self._schema.fields])

    def get_constructor(self):
      def constructor(values):
        return self._type(**values)
      return constructor


if sys.version_info[:2] >= (3, 7):
  import dataclasses

  @SchemaConverter.register
  class DataclassSchemaConverter(SchemaConverter):
    @classmethod
    def claim_type(cls, type_):
      return dataclasses.is_dataclass(type_)

    def get_fields(self):
      return [
          schema_pb2.Field(
              name=field.name, type=typing_to_runner_api(field.type))
          for field in dataclasses.fields(self._type)]

    def get_field_values(self, instance):
      return [getattr(instance, f.name) for f in self.schema.fields]

    def make_type(self, type_name):
      return dataclasses.make_dataclass(
          type_name,
          [(field.name, typing_from_runner_api(field.type))
           for field in self._schema.fields])

    def get_constructor(self):
      def constructor(values):
        return self._type(**values)

      return constructor


@SchemaConverter.register
class AttrsConverter(SchemaConverter):
  @classmethod
  def claim_type(cls, type_):
    attr = sys.modules.get('attr')
    if attr:
      try:
        attr.fields(type_)
      except (attr.exceptions.NotAnAttrsClassError, TypeError):
        return False
      else:
        return True
    return False

  def get_fields(self):
    import attr
    return [
        schema_pb2.Field(
            name=field.name, type=typing_to_runner_api(field.type))
        for field in attr.fields(self._type)]

  def get_field_values(self, instance):
    return [getattr(instance, f.name) for f in self.schema.fields]

  def make_type(self, type_name):
    import attr
    return attr.make_class(
        type_name,
        OrderedDict(
            [(field.name, attr.ib(type=typing_from_runner_api(field.type)))
             for field in self._schema.fields]))

  def get_constructor(self):
    def constructor(values):
      return self._type(**values)

    return constructor


def converter_from_schema(schema):
  # type: (schema_pb2.Schema) -> SchemaConverter
  converter = SCHEMA_REGISTRY.get_by_schema(schema)
  if converter is None:
    converter = NamedTupleSchemaConverter.from_schema(schema)
    SCHEMA_REGISTRY.add(converter)
  return converter


def type_from_schema(schema):
  # type: (schema_pb2.Schema) -> Type
  return converter_from_schema(schema).type


def type_to_schema(type_):
  # type: (Type) -> schema_pb2.Schema
  return typing_to_runner_api(type_).row_type.schema


def register_schema(type_):
  """Enable the use of the RowCoder with this type.

  Validates that the type can be converted to a schema.

  Can be used as a decorator.
  """
  typing_to_runner_api(type_)
  return type_
