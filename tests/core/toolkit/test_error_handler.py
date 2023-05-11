# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections.abc import Iterable
import os
import tempfile

# Third Party
import pytest

# First Party
import alog

# Local
from caikit.core.toolkit.errors import error_handler

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit


class TestErrorHandler(TestCaseBase):
    def setUp(self):
        """Construct a new log channel and error handler."""
        self.log = alog.use_channel("TEST_ERR_HAN")
        self.error = error_handler.get(self.log)

    @pytest.fixture(autouse=True)
    def caplog(self, caplog):
        self.caplog = caplog

    def test_reuse_chan(self):
        """Verify that we are reusing the same channal when called with the same name."""
        for i in range(3):
            error = error_handler.get(self.log)
            self.assertTrue(error is self.error)

    def test_call_directly(self):
        """Verify that calling an error handler is equivalent to calling the `.log_raises` method."""
        self.assertTrue(
            error_handler.ErrorHandler.log_raise is error_handler.ErrorHandler.__call__
        )

    def test_log_raises_raises(self):
        """Verify that `log_raises` and `__call__` actually raises an exception."""
        with self.assertRaises(ValueError):
            self.error.log_raise("<XYZ>", ValueError("this is a test"))

        with self.assertRaises(FileNotFoundError):
            self.error("<YYY>", FileNotFoundError("this is a test"))

        with self.assertRaises(ValueError):
            self.error.log_raise(
                "<XYZ>", ValueError("this is a test"), RuntimeError("root error")
            )

        with self.assertRaises(ValueError):
            self.error(
                "<YYY>", ValueError("this is a test"), RuntimeError("root error")
            )

    def test_log_raises_logs(self):
        """Verify that `log_raises` actually logs an error message."""
        try:
            self.error("<AAA>", TypeError("this is a test"))
        except:
            pass

        # verify that one log line was omitted
        self.assertEqual(len(self.caplog.records), 1)

    def test_log_raises_max(self):
        """Verify that `log_raises` will not keep logging after max log lines written."""
        # raise the same exception over and over, max + 10 times
        exception = AttributeError("this is a test")
        for i in range(caikit.get_config().max_exception_log_messages + 10):
            try:
                self.error("<BBB>", exception)
            except:
                pass

        # verify that tracking attribute was set
        self.assertTrue(hasattr(exception, "_caikit_core_nexception_log_messages"))
        # verify that it was incremented for every raise
        self.assertEqual(
            exception._caikit_core_nexception_log_messages,
            caikit.get_config().max_exception_log_messages + 10 - 1,
        )
        # verify that it only wrote the max log lines
        # one for extra for message that logging will stop
        self.assertEqual(
            len(self.caplog.records), caikit.get_config().max_exception_log_messages + 1
        )

    def test_log_raises_max_with_root(self):
        """Verify that `log_raises` will not keep logging after max log lines written."""
        # raise the same exception over and over, max + 10 times
        exception = AttributeError("this is a test")
        root_exception = ValueError("this is the root ex")
        for i in range(caikit.get_config().max_exception_log_messages + 10):
            try:
                self.error("<BBB>", exception, root_exception)
            except:
                pass

        # verify that tracking attribute was set
        self.assertTrue(hasattr(exception, "_caikit_core_nexception_log_messages"))
        self.assertTrue(hasattr(root_exception, "_caikit_core_nexception_log_messages"))
        # verify that it was incremented for every raise
        self.assertEqual(
            exception._caikit_core_nexception_log_messages,
            caikit.get_config().max_exception_log_messages + 10 - 1,
        )
        self.assertEqual(
            root_exception._caikit_core_nexception_log_messages,
            caikit.get_config().max_exception_log_messages + 10 - 1,
        )

        # verify that it only wrote the max log lines
        # one for extra for message that logging will stop
        # Length of all records == 2 * MAX (one for each: exception and root exception)
        self.assertEqual(
            len(self.caplog.records),
            2 * (caikit.get_config().max_exception_log_messages + 1),
        )

    def test_type_check_raises(self):
        """Verify that `type_check` raises a `TypeError`."""
        with self.assertRaises(TypeError):
            self.error.type_check("<HHH>", int, float, bad_type="hello world")

    def test_type_check_passes(self):
        """Verify that `type_check` does not raise an exception when good."""
        try:
            self.error.type_check("<III>", int, float, good_type=42)
        except:
            self.assertTrue(False)

        try:
            self.error.type_check("<III>", int, float, good_type=42, another=33)
        except:
            self.assertTrue(False)

        try:
            self.error.type_check("<III>", int, str, good_type=42, another="abc")
        except:
            self.assertTrue(False)

    def test_type_check_none(self):
        """Verify that `type_check` does not raise when `None` values are allowed."""
        try:
            self.error.type_check(
                "<III>", str, allow_none=True, good_type=None, another="abc"
            )
        except:
            self.assertTrue(False)

    def test_type_check_logs(self):
        """Verify that `type_check` writes a single log line."""
        with self.assertRaises(TypeError):
            self.error.type_check("<JJJ>", int, bad_type="hello world")

        self.assertEqual(len(self.caplog.records), 1)

    def test_type_check_all_raises(self):
        """Verify that `type_check_all` raises a `TypeError`."""
        with self.assertRaises(TypeError):
            self.error.type_check_all("<ABC>", int, bad_type=(1, 2, "x"))

    def test_type_check_all_passes(self):
        """Verify that `type_check_all` does not reaise an exception when good."""
        try:
            self.error.type_check_all(
                "<III>", int, str, good_type=(42, "abc", "abc", 33)
            )
        except:
            self.assertTrue(False)

        try:
            self.error.type_check_all(
                "<III>",
                int,
                str,
                good_type=(
                    42,
                    "abc",
                ),
                another=[33, 33],
            )
        except:
            self.assertTrue(False)

    def test_type_check_all_iterable(self):
        """Verify that a custom Iterable passes type_check_all"""

        class CustomIterable(Iterable):
            def __init__(self, lst):
                self._lst = lst

            def __iter__(self):
                return self._lst.__iter__()

        self.error.type_check_all("<III>", int, good_type=CustomIterable([1, 2, 3]))

    def test_type_check_all_str(self):
        """Verify that a top level str object fails type_check_all"""
        with self.assertRaises(TypeError):
            self.error.type_check_all("<III>", str, str_type="<test_str>")

    def test_type_check_all_generator(self):
        """Verify that a top level generator object fails type_check_all"""
        with self.assertRaises(TypeError):
            self.error.type_check_all(
                "<III>", int, generator_type=(n for n in range(5))
            )

    def test_type_check_empty_passes(self):
        """Verify that `type_check_all` does not raise for empty variables."""
        try:
            self.error.type_check_all("<III>", int, str, good_type=(), another=[])
        except:
            self.assertTrue(False)

        try:
            self.error.type_check_all(
                "<III>", int, None, int, allow_none=True, good_type=(), another=[]
            )
        except:
            self.assertTrue(False)

        try:
            self.error.type_check_all(
                "<III>", int, good_type=(), another=[], again=[1, 2, 3]
            )
        except:
            self.assertTrue(False)

    def test_type_check_no_types(self):
        """Verify that a `RuntimeError` is raised if no types are passed."""
        with self.assertRaises(RuntimeError):
            self.error.type_check("<ABC>", foo="abc", bar=33)

    def test_type_check_no_variables(self):
        """Verify that a `RuntimeError` is raised if no variables are passed."""
        with self.assertRaises(RuntimeError):
            self.error.type_check("<ABC>", int, str, allow_none=True)

        with self.assertRaises(RuntimeError):
            self.error.type_check("<ABC>", int, str, allow_none=False)

    def test_type_check_all_non_list_or_tuple(self):
        """Verify that `type_check_all` raises a `TypeError` if variable is not a list or tuple.."""
        with self.assertRaises(TypeError):
            self.error.type_check_all("<ABC>", int, float, foo="abc")

        with self.assertRaises(TypeError):
            self.error.type_check_all("<ABC>", int, float, foo=3)

        with self.assertRaises(TypeError):
            self.error.type_check_all("<ABC>", int, float, foo={"a": 1, "b": 2})

    def test_type_check_all_logs(self):
        """Verify that `type_check_all` omitts a log line."""
        with self.assertRaises(TypeError):
            self.error.type_check_all("<ABC>", int, bad_type=(3.3, "abc"))

        self.assertEqual(len(self.caplog.records), 1)

    def test_type_check_all_none(self):
        """Verify that `type_check_all` allows `None` values, but not to contain `None` values."""
        try:
            self.error.type_check_all(
                "<XYZ>", int, allow_none=True, foo=(1, 2, 3), bar=None
            )
        except:
            self.assertTrue(False)

        with self.assertRaises(TypeError):
            self.error.type_check_all(
                "<XYZ>", int, allow_none=True, foo=(1, 2, None), bar=None
            )

    def test_type_check_all_no_types(self):
        """Verify that a `RuntimeError` is raised if no types are passed."""
        with self.assertRaises(RuntimeError):
            self.error.type_check_all("<ABC>", foo=(1, 2, 3), bar=("a", "b", "c"))

    def test_type_all_check_no_variables(self):
        """Verify that a `RuntimeError` is raised if no variables are passed."""
        with self.assertRaises(RuntimeError):
            self.error.type_check_all("<ABC>", int, str, allow_none=True)

        with self.assertRaises(RuntimeError):
            self.error.type_check_all("<ABC>", int, str, allow_none=False)

    def test_subclass_check_raises(self):
        """Verify that `subclass_check` raises a `TypeError`."""
        with pytest.raises(TypeError):
            self.error.subclass_check("<HHH>", int, float)

    def test_subclass_check_passes(self):
        """Verify that `subclass_check` does not raise an exception when good."""

        class Base:
            pass

        class Derived(Base):
            pass

        class SecondOrder(Derived):
            pass

        class Other:
            pass

        self.error.subclass_check("<III>", Derived, Base)
        self.error.subclass_check("<III>", SecondOrder, Base)
        self.error.subclass_check("<III>", Derived, Base, Other)
        self.error.subclass_check("<III>", None, Base, Other, allow_none=True)

    def test_subclass_check_none(self):
        """Verify that `subclass_check` does not raise when `None` values are allowed."""
        self.error.subclass_check("<III>", None, int, allow_none=True)

    def test_subclass_check_logs(self):
        """Verify that `subclass_check` writes a single log line."""
        with pytest.raises(TypeError):
            self.error.subclass_check("<JJJ>", int, float)

        self.assertEqual(len(self.caplog.records), 1)

    def test_subclass_requires_base_type(self):
        """Verify that `subclass_check` raises a RuntimeError if no base given"""
        with pytest.raises(RuntimeError):
            self.error.subclass_check("<JJJ>", int)

    def test_subclass_check_non_type(self):
        """Verify that `subclass_check` raises if the value is not a type"""
        with pytest.raises(TypeError):
            self.error.subclass_check("<III>", 1, int, allow_none=True)

    def test_value_check_raises(self):
        """Verify that `value_check` raises a `ValueError`."""
        bad_value = 1.1

        with self.assertRaises(ValueError):
            self.error.value_check(
                "<CCC>", 0.0 <= bad_value <= 1.0, "invalid range `{}`", bad_value
            )
        error_msg = self.caplog.records[0].msg["message"]
        self.assertEqual(
            error_msg,
            "exception raised: ValueError('value check failed: invalid range `1.1`')",
        )

    def test_value_check_no_args_raises(self):
        """Verify that `value_check` raises a `ValueError` if no args are given with empty msg."""
        bad_value = 1.1

        with self.assertRaises(ValueError):
            self.error.value_check("<CCC>", 0.0 <= bad_value <= 1.0)
        error_msg = self.caplog.records[0].msg["message"]
        self.assertEqual(
            error_msg,
            "exception raised: ValueError('value check failed: ')",
        )

    def test_value_check_empty_msg_raises(self):
        """Verify that `value_check` raises a `ValueError` if msg is empty."""
        bad_value = 1.1

        with self.assertRaises(ValueError):
            self.error.value_check("<CCC>", 0.0 <= bad_value <= 1.0, bad_value)
        error_msg = self.caplog.records[0].msg["message"]
        self.assertEqual(
            error_msg,
            "exception raised: ValueError('value check failed: 1.1')",
        )

    def test_value_check_empty_args_raises(self):
        """Verify that `value_check` raises a `ValueError` if only msg is provided."""
        bad_value = 1.1

        with self.assertRaises(ValueError):
            self.error.value_check("<CCC>", 0.0 <= bad_value <= 1.0, "invalid range")
        error_msg = self.caplog.records[0].msg["message"]
        self.assertEqual(
            error_msg,
            "exception raised: ValueError('value check failed: invalid range')",
        )

    def test_value_check_passes(self):
        """Verify that `value_check` does not raise an exception when good."""
        good_value = 0.5
        try:
            self.error.value_check(
                "<DDD>", 0.0 <= good_value <= 1.0, "invalid range `{}`", good_value
            )
        except:
            self.assertTrue(False)

    def test_value_check_logs(self):
        """Verify that `value_check` writes a single log line."""
        with self.assertRaises(ValueError):
            self.error.value_check("<EEE>", False, "invalid range")

        self.assertEqual(len(self.caplog.records), 1)

    def test_file_check_raises(self):
        """Verify that `file_check` fails when file is not found."""
        with self.assertRaises(FileNotFoundError):
            self.error.file_check(
                "<ABC>", "/etc/", "/hopefully/not/a/file/path", "/tmp"
            )

        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(FileNotFoundError):
                self.error.file_check("<123>", tempdir)

    def test_file_check_logs(self):
        """Verify that `file_check` omitts a log message."""
        with self.assertRaises(FileNotFoundError):
            self.error.file_check("<XYZ>", "/definitely/not/a/valid/file/path")

        self.assertEqual(len(self.caplog.records), 1)

    def test_file_check_passes(self):
        """Verify that `file_check` does not raise when the file exists."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_filename = os.path.join(tempdir, "test.txt")

            with open(temp_filename, mode="w") as fh:
                fh.write("hello world")

            self.assertTrue(
                os.path.exists(temp_filename) and os.path.isfile(temp_filename)
            )

            try:
                self.error.file_check("<ABC>", temp_filename)
            except:
                self.assertTrue(False)

    def test_dir_check_raises(self):
        """Verify that `dir_check` fails when directory is not found."""
        with self.assertRaises(FileNotFoundError):
            self.error.dir_check("<ABC>", "/etc/", "/hopefully/not/a/dir/path", "/tmp")

    def test_dir_check_raises_on_file(self):
        """Verify that `dir_check` fails when pointed to a file."""
        with tempfile.TemporaryDirectory() as tempdir:
            temp_filename = os.path.join(tempdir, "test.txt")

            with open(temp_filename, mode="w") as fh:
                fh.write("hello world")

            self.assertTrue(
                os.path.exists(temp_filename) and os.path.isfile(temp_filename)
            )

            with self.assertRaises(NotADirectoryError):
                self.error.dir_check("<ABC>", temp_filename)

    def test_dir_check_logs(self):
        """Verify that `dir_check` omitts a log message."""
        with self.assertRaises(FileNotFoundError):
            self.error.dir_check("<XYZ>", "/definitely/not/a/valid/dir/path")

        self.assertEqual(len(self.caplog.records), 1)

    def test_dir_check_passes(self):
        """Verify that `dir_check` does not raise when the directory exists."""
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                self.error.dir_check("<ABC>", tempdir)
            except:
                self.assertTrue(False)
