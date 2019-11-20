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
"""Tests for schemas."""

from __future__ import absolute_import

from collections import OrderedDict
import itertools
import sys
import unittest
from typing import ByteString
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import attr
import numpy as np
from mypy_extensions import TypedDict
from past.builtins import unicode
from mock import patch

from apache_beam import coders
from apache_beam.coders import typecoders
from apache_beam.portability.api import schema_pb2
from apache_beam.typehints.native_type_compatibility import _match_is_named_tuple
from apache_beam.typehints.schemas import typing_from_runner_api
from apache_beam.typehints.schemas import typing_to_runner_api

IS_PYTHON_3 = sys.version_info.major > 2


def primitive_types():
  all_nonoptional_primitives = [
      np.int8,
      np.int16,
      np.int32,
      np.int64,
      np.float32,
      np.float64,
      unicode,
      bool,
  ]

  # The bytes type cannot survive a roundtrip to/from proto in Python 2.
  # In order to use BYTES a user type has to use typing.ByteString (because
  # bytes == str, and we map str to STRING).
  if IS_PYTHON_3:
    all_nonoptional_primitives.extend([bytes])

  all_optional_primitives = [
      Optional[typ] for typ in all_nonoptional_primitives
  ]

  return all_nonoptional_primitives + all_optional_primitives


def primitive_fields():
  all_nonoptional_primitives = [
      schema_pb2.FieldType(atomic_type=typ)
      for typ in schema_pb2.AtomicType.values()
      if typ is not schema_pb2.UNSPECIFIED
  ]

  # The bytes type cannot survive a roundtrip to/from proto in Python 2.
  # In order to use BYTES a user type has to use typing.ByteString (because
  # bytes == str, and we map str to STRING).
  if not IS_PYTHON_3:
    all_nonoptional_primitives.remove(
        schema_pb2.FieldType(atomic_type=schema_pb2.BYTES))

  all_optional_primitives = [
      schema_pb2.FieldType(nullable=True, atomic_type=typ)
      for typ in schema_pb2.AtomicType.values()
      if typ is not schema_pb2.UNSPECIFIED
  ]

  return all_nonoptional_primitives + all_optional_primitives


class FieldTypeTest(unittest.TestCase):
  """ Tests for Runner API FieldType proto to/from typing conversions

  There are two main tests: test_typing_survives_proto_roundtrip, and
  test_proto_survives_typing_roundtrip. These are both necessary because Schemas
  are cached by ID, so performing just one of them wouldn't necessarily exercise
  all code paths.
  """

  def test_typing_survives_proto_roundtrip(self):
    all_primitives = primitive_types()
    basic_array_types = [Sequence[typ] for typ in all_primitives]

    basic_map_types = [
        Mapping[key_type,
                value_type] for key_type, value_type in itertools.product(
                    all_primitives, all_primitives)
    ]

    test_cases = all_primitives + \
                 basic_array_types + \
                 basic_map_types

    for test_case in test_cases:
      self.assertEqual(test_case,
                       typing_from_runner_api(typing_to_runner_api(test_case)))

  def test_field_proto_survives_typing_roundtrip(self):
    all_primitives = primitive_fields()
    basic_array_types = [
        schema_pb2.FieldType(array_type=schema_pb2.ArrayType(element_type=typ))
        for typ in all_primitives
    ]

    basic_map_types = [
        schema_pb2.FieldType(
            map_type=schema_pb2.MapType(
                key_type=key_type, value_type=value_type)) for key_type,
        value_type in itertools.product(all_primitives, all_primitives)
    ]

    test_cases = all_primitives + \
                 basic_array_types + \
                 basic_map_types

    for test_case in test_cases:
      self.assertEqual(test_case,
                       typing_to_runner_api(typing_from_runner_api(test_case)))

  def test_schema_proto_survives_typing_roundtrip(self):
    all_primitives = primitive_fields()
    selected_schemas = [
        schema_pb2.FieldType(
            row_type=schema_pb2.RowType(
                schema=schema_pb2.Schema(
                    id='32497414-85e8-46b7-9c90-9a9cc62fe390',
                    fields=[
                        schema_pb2.Field(name='field%d' % i, type=typ)
                        for i, typ in enumerate(all_primitives)
                    ]))),
        schema_pb2.FieldType(
            row_type=schema_pb2.RowType(
                schema=schema_pb2.Schema(
                    id='dead1637-3204-4bcb-acf8-99675f338600',
                    fields=[
                        schema_pb2.Field(
                            name='id',
                            type=schema_pb2.FieldType(
                                atomic_type=schema_pb2.INT64)),
                        schema_pb2.Field(
                            name='name',
                            type=schema_pb2.FieldType(
                                atomic_type=schema_pb2.STRING)),
                        schema_pb2.Field(
                            name='optional_map',
                            type=schema_pb2.FieldType(
                                nullable=True,
                                map_type=schema_pb2.MapType(
                                    key_type=schema_pb2.FieldType(
                                        atomic_type=schema_pb2.STRING
                                    ),
                                    value_type=schema_pb2.FieldType(
                                        atomic_type=schema_pb2.DOUBLE
                                    )))),
                        schema_pb2.Field(
                            name='optional_array',
                            type=schema_pb2.FieldType(
                                nullable=True,
                                array_type=schema_pb2.ArrayType(
                                    element_type=schema_pb2.FieldType(
                                        atomic_type=schema_pb2.FLOAT)
                                ))),
                        schema_pb2.Field(
                            name='array_optional',
                            type=schema_pb2.FieldType(
                                array_type=schema_pb2.ArrayType(
                                    element_type=schema_pb2.FieldType(
                                        nullable=True,
                                        atomic_type=schema_pb2.BYTES)
                                ))),
                    ]))),
    ]

    for test_schema in selected_schemas:
      with patch.object(coders, 'registry',
                        typecoders.CoderRegistry()) as mock_registry:
        typing = typing_from_runner_api(test_schema)
        # type should be registered
        self.assertEqual([typing], mock_registry.custom_types)
        self.assertEqual(getattr(typing, '__beam_schema_id__'),
                         test_schema.row_type.schema.id)
        self.assertTrue(_match_is_named_tuple(typing))
        self.assertEqual(test_schema, typing_to_runner_api(typing))
        # nothing new should be registered
        self.assertEqual([typing], mock_registry.custom_types)

  def test_unknown_primitive_raise_valueerror(self):
    self.assertRaises(ValueError, lambda: typing_to_runner_api(np.uint32))

  def test_unknown_atomic_raise_valueerror(self):
    self.assertRaises(
        ValueError, lambda: typing_from_runner_api(
            schema_pb2.FieldType(atomic_type=schema_pb2.UNSPECIFIED))
    )

  @unittest.skipIf(IS_PYTHON_3, 'str is acceptable in python 3')
  def test_str_raises_error_py2(self):
    self.assertRaises(lambda: typing_to_runner_api(str))

  def test_int_maps_to_int64(self):
    self.assertEqual(
        schema_pb2.FieldType(atomic_type=schema_pb2.INT64),
        typing_to_runner_api(int))

  def test_float_maps_to_float64(self):
    self.assertEqual(
        schema_pb2.FieldType(atomic_type=schema_pb2.DOUBLE),
        typing_to_runner_api(float))

  @unittest.skipIf(IS_PYTHON_3, 'str is acceptable in python 3')
  def test_str_raises_error_py2(self):
    self.assertRaises(lambda: typing_to_runner_api(
        NamedTuple('Test', [('int', int), ('str', str)])))


class SchemaTestMixin(object):
  """ Tests for Runner API Schema proto to/from typing conversions

  There are two main tests: test_typing_survives_proto_roundtrip, and
  test_proto_survives_typing_roundtrip. These are both necessary because Schemas
  are cached by ID, so performing just one of them wouldn't necessarily exercise
  all code paths.
  """

  def get_type_with_primitive_fields(self):
    raise NotImplementedError

  def get_complex_type(self):
    raise NotImplementedError

  def get_person_type(self):
    raise NotImplementedError

  def test_typing_survives_proto_roundtrip(self):
    selected_types = [
        self.get_type_with_primitive_fields(),
        self.get_complex_type(),
    ]

    for test_type, child_types in selected_types:
      with patch.object(coders, 'registry',
                        typecoders.CoderRegistry()) as mock_registry:
        self.assertFalse(hasattr(test_type, '__beam_schema_id__'))
        schema = typing_to_runner_api(test_type)
        # type should be registered (child types will also be present)
        self.assertEqual(child_types + [test_type], mock_registry.custom_types)
        self.assertEqual(getattr(test_type, '__beam_schema_id__'),
                         schema.row_type.schema.id)
        self.assertIs(test_type, typing_from_runner_api(schema))
        # nothing new should be registered
        self.assertEqual(child_types + [test_type], mock_registry.custom_types)

  def test_trivial_example(self):
    test_class = self.get_person_type()

    expected = schema_pb2.FieldType(
        row_type=schema_pb2.RowType(
            schema=schema_pb2.Schema(fields=[
                schema_pb2.Field(
                    name='name',
                    type=schema_pb2.FieldType(
                        atomic_type=schema_pb2.STRING),
                ),
                schema_pb2.Field(
                    name='age',
                    type=schema_pb2.FieldType(
                        nullable=True,
                        atomic_type=schema_pb2.INT64)),
                schema_pb2.Field(
                    name='interests',
                    type=schema_pb2.FieldType(
                        array_type=schema_pb2.ArrayType(
                            element_type=schema_pb2.FieldType(
                                atomic_type=schema_pb2.STRING)))),
                schema_pb2.Field(
                    name='height',
                    type=schema_pb2.FieldType(
                        atomic_type=schema_pb2.DOUBLE)),
                schema_pb2.Field(
                    name='blob',
                    type=schema_pb2.FieldType(
                        atomic_type=schema_pb2.BYTES)),
            ])))

    # Only test that the fields are equal. If we attempt to test the entire type
    # or the entire schema, the generated id will break equality.
    with patch.object(coders, 'registry',
                      typecoders.CoderRegistry()) as mock_registry:
      schema = typing_to_runner_api(test_class)
      self.assertEqual(expected.row_type.schema.fields,
                       schema.row_type.schema.fields)


class NamedTupleSchemaTest(unittest.TestCase, SchemaTestMixin):
  def get_type_with_primitive_fields(self):
    schema_type = NamedTuple('AllPrimitives', [
        ('field%d' % i, typ) for i, typ in enumerate(primitive_types())
    ])
    return schema_type, []

  def get_complex_type(self):
    child_type = NamedTuple('ChildSchema', [('name', unicode)])

    schema_type = NamedTuple('ComplexSchema', [
        ('id', np.int64),
        ('name', unicode),
        ('optional_map', Optional[Mapping[unicode,
                                          Optional[np.float64]]]),
        ('optional_array', Optional[Sequence[np.float32]]),
        ('array_optional', Sequence[Optional[bool]]),
        ('nested_schema', Sequence[child_type]),
    ])
    return schema_type, [child_type]

  def get_person_type(self):
    return NamedTuple('Person', [
        ('name', unicode),
        ('age', Optional[int]),
        ('interests', List[unicode]),
        ('height', float),
        ('blob', ByteString),
    ])


class TypedDictSchemaTest(unittest.TestCase, SchemaTestMixin):
  def get_type_with_primitive_fields(self):
    schema_type =  TypedDict('AllPrimitives', [
        ('field%d' % i, typ) for i, typ in enumerate(primitive_types())
    ])
    return schema_type, []

  def get_complex_type(self):
    child_type = TypedDict('ChildSchema', [('name', unicode)])

    schema_type = TypedDict('ComplexSchema', [
        ('id', np.int64),
        ('name', unicode),
        ('optional_map', Optional[Mapping[unicode,
                                          Optional[np.float64]]]),
        ('optional_array', Optional[Sequence[np.float32]]),
        ('nested_schema', Sequence[child_type]),
    ])
    return schema_type, [child_type]

  def get_person_type(self):
    return TypedDict('Person', [
        ('name', unicode),
        ('age', Optional[int]),
        ('interests', List[unicode]),
        ('height', float),
        ('blob', ByteString),
    ])


class AttrsSchemaTest(unittest.TestCase, SchemaTestMixin):
  def get_type_with_primitive_fields(self):
    schema_type = attr.make_class(
        'AllPrimitives',
        OrderedDict(
            ('field%d' % i, attr.ib(type=typ))
            for i, typ in enumerate(primitive_types())))
    return schema_type, []

  def get_complex_type(self):
    @attr.s
    class ChildSchema(object):
      name = attr.ib(type=unicode)

    @attr.s
    class ComplexSchema(object):
      id = attr.ib(type=np.int64)
      name = attr.ib(type=unicode)
      optional_map = attr.ib(type=Optional[Mapping[unicode,
                                                   Optional[np.float64]]])
      optional_array = attr.ib(type=Optional[Sequence[np.float32]])
      nested_schema = attr.ib(type=Sequence[ChildSchema])

    return ComplexSchema, [ChildSchema]

  def get_person_type(self):
    @attr.s
    class Person(object):
      name = attr.ib(type=unicode)
      age = attr.ib(type=Optional[int])
      interests = attr.ib(type=List[unicode])
      height = attr.ib(type=float)
      blob = attr.ib(type=ByteString)

    return Person


if __name__ == '__main__':
  unittest.main()
