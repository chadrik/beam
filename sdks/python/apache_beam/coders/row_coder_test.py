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
from __future__ import absolute_import

import logging
import typing
import unittest
from contextlib import contextmanager
from itertools import chain

import attr
import numpy as np
from mypy_extensions import TypedDict
from past.builtins import unicode
from mock import patch

from apache_beam import coders
from apache_beam.coders import typecoders
from apache_beam.coders import RowCoder
from apache_beam.portability.api import schema_pb2
from apache_beam.typehints.schemas import typing_to_runner_api


class RowCoderTestMixin(object):

  def get_test_cases(self):
    raise NotImplementedError

  def get_person_type(self):
    raise NotImplementedError

  @contextmanager
  def person_registry(self):
    Person = self.get_person_type()
    with patch.object(coders, 'registry',
                      typecoders.CoderRegistry()) as mock_registry:
      mock_registry.register_coder(Person, RowCoder)
      yield mock_registry, Person

  def test_create_row_coder_from_type(self):

    Person = self.get_person_type()

    with patch.object(coders, 'registry',
                      typecoders.CoderRegistry()) as mock_registry:
      mock_registry.register_coder(Person, RowCoder)

      expected_coder = RowCoder(typing_to_runner_api(Person).row_type.schema)
      real_coder = mock_registry.get_coder(Person)

      for test_case in self.get_test_cases():
        self.assertEqual(
            expected_coder.encode(test_case), real_coder.encode(test_case))

        self.assertEqual(test_case,
                         real_coder.decode(real_coder.encode(test_case)))

  def test_create_row_coder_from_schema(self):

    schema = schema_pb2.Schema(
        id="person",
        fields=[
            schema_pb2.Field(
                name="name",
                type=schema_pb2.FieldType(
                    atomic_type=schema_pb2.STRING)),
            schema_pb2.Field(
                name="age",
                type=schema_pb2.FieldType(
                    atomic_type=schema_pb2.INT32)),
            schema_pb2.Field(
                name="address",
                type=schema_pb2.FieldType(
                    atomic_type=schema_pb2.STRING, nullable=True)),
            schema_pb2.Field(
                name="aliases",
                type=schema_pb2.FieldType(
                    array_type=schema_pb2.ArrayType(
                        element_type=schema_pb2.FieldType(
                            atomic_type=schema_pb2.STRING)))),
        ])

    Person = self.get_person_type()

    with patch.object(coders, 'registry',
                      typecoders.CoderRegistry()) as mock_registry:
      mock_registry.register_coder(Person, RowCoder)
      # type should be registered
      self.assertEqual([Person], mock_registry.custom_types)

      coder = RowCoder(schema)

      for test_case in self.get_test_cases():
        result = coder.decode(coder.encode(test_case))
        self.assertEqual(test_case, result)
        self.assertIs(type(test_case), type(result))

      # nothing new should be registered
      self.assertEqual([Person], mock_registry.custom_types)

    # repeat without explicit registration
    with patch.object(coders, 'registry',
                      typecoders.CoderRegistry()) as mock_registry:

      # type should not be registered
      self.assertEqual([], mock_registry.custom_types)

      coder = RowCoder(schema)

      for test_case in self.get_test_cases():
        self.assertEqual(test_case, coder.decode(coder.encode(test_case)))

      # nothing new should be registered
      self.assertEqual([], mock_registry.custom_types)


class NamedTupleRowCoderTest(unittest.TestCase, RowCoderTestMixin):
  Person = typing.NamedTuple("Person", [
      ("name", unicode),
      ("age", np.int32),
      ("address", typing.Optional[unicode]),
      ("aliases", typing.List[unicode]),
  ])

  def get_test_cases(self):
    return [
        self.Person("Jon Snow", 23, None, ["crow", "wildling"]),
        self.Person("Daenerys Targaryen", 25, "Westeros", ["Mother of Dragons"]),
        self.Person("Michael Bluth", 30, None, [])
    ]

  def get_person_type(self):
    return self.Person

  @unittest.skip(
      "BEAM-8030 - Overflow behavior in VarIntCoder is currently inconsistent"
  )
  def test_overflows(self):
    IntTester = typing.NamedTuple('IntTester', [
        # TODO(BEAM-7996): Test int8 and int16 here as well when those types are
        # supported
        # ('i8', typing.Optional[np.int8]),
        # ('i16', typing.Optional[np.int16]),
        ('i32', typing.Optional[np.int32]),
        ('i64', typing.Optional[np.int64]),
    ])

    c = RowCoder.from_type_hint(IntTester, None)

    no_overflow = chain(
        (IntTester(i32=i, i64=None) for i in (-2**31, 2**31-1)),
        (IntTester(i32=None, i64=i) for i in (-2**63, 2**63-1)),
    )

    # Encode max/min ints to make sure they don't throw any error
    for case in no_overflow:
      c.encode(case)

    overflow = chain(
        (IntTester(i32=i, i64=None) for i in (-2**31-1, 2**31)),
        (IntTester(i32=None, i64=i) for i in (-2**63-1, 2**63)),
    )

    # Encode max+1/min-1 ints to make sure they DO throw an error
    for case in overflow:
      self.assertRaises(OverflowError, lambda: c.encode(case))

  def test_none_in_non_nullable_field_throws(self):
    Test = typing.NamedTuple('Test', [('foo', unicode)])

    c = RowCoder.from_type_hint(Test, None)
    self.assertRaises(ValueError, lambda: c.encode(Test(foo=None)))

  def test_schema_remove_column(self):
    fields = [("field1", unicode), ("field2", unicode)]
    # new schema is missing one field that was in the old schema
    Old = typing.NamedTuple('Old', fields)
    New = typing.NamedTuple('New', fields[:-1])

    old_coder = RowCoder.from_type_hint(Old, None)
    new_coder = RowCoder.from_type_hint(New, None)

    self.assertEqual(
        New("foo"), new_coder.decode(old_coder.encode(Old("foo", "bar"))))

  def test_schema_add_column(self):
    fields = [("field1", unicode), ("field2", typing.Optional[unicode])]
    # new schema has one (optional) field that didn't exist in the old schema
    Old = typing.NamedTuple('Old', fields[:-1])
    New = typing.NamedTuple('New', fields)

    old_coder = RowCoder.from_type_hint(Old, None)
    new_coder = RowCoder.from_type_hint(New, None)

    self.assertEqual(
        New("bar", None), new_coder.decode(old_coder.encode(Old("bar"))))

  def test_schema_add_column_with_null_value(self):
    fields = [("field1", typing.Optional[unicode]), ("field2", unicode),
              ("field3", typing.Optional[unicode])]
    # new schema has one (optional) field that didn't exist in the old schema
    Old = typing.NamedTuple('Old', fields[:-1])
    New = typing.NamedTuple('New', fields)

    old_coder = RowCoder.from_type_hint(Old, None)
    new_coder = RowCoder.from_type_hint(New, None)

    self.assertEqual(
        New(None, "baz", None),
        new_coder.decode(old_coder.encode(Old(None, "baz"))))


class TypedDictRowCoderTest(unittest.TestCase, RowCoderTestMixin):

  Person = TypedDict("Person", [
      ("name", unicode),
      ("age", np.int32),
      ("address", typing.Optional[unicode]),
      ("aliases", typing.List[unicode]),
  ])

  def get_person_type(self):
    return self.Person

  def get_test_cases(self):
    Person = self.get_person_type()
    return [
        Person(name="Jon Snow", age=23, address=None,
               aliases=["crow", "wildling"]),
        Person(name="Daenerys Targaryen", age=25, address="Westeros",
               aliases=["Mother of Dragons"]),
        Person(name="Michael Bluth", age=30, address=None, aliases=[])
    ]


class AttrstRowCoderTest(unittest.TestCase, RowCoderTestMixin):

  @attr.s
  class Person(object):
    name = attr.ib(type=unicode)
    age = attr.ib(type=np.int32)
    address = attr.ib(type=typing.Optional[unicode])
    aliases = attr.ib(type=typing.List[unicode])

  def get_person_type(self):
    return self.Person

  def get_test_cases(self):
    Person = self.get_person_type()
    return [
        Person(name="Jon Snow", age=23, address=None,
               aliases=["crow", "wildling"]),
        Person(name="Daenerys Targaryen", age=25, address="Westeros",
               aliases=["Mother of Dragons"]),
        Person(name="Michael Bluth", age=30, address=None, aliases=[])
    ]


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
