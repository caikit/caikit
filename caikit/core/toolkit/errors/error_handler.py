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


"""Custom exception and error handling logic.
"""

# Standard
from collections.abc import Iterable
from types import GeneratorType
from typing import Any
import os

# Local
from caikit.config import get_config

# dictionary mapping string log channel name to error handler instances
# there is only one error handler instance for each log channel name
_error_handlers = {}


def get(log_chan):
    """Get an error handler associated with a given alog log channel.  The same error handler will
    be returned if this function is called repeatedly with the same log channel.

    Args:
        name: alog channel
            An alog log channel.

    Returns:
        ErrorHandler: An instance of `ErrorHandler` associated with `log_chan`
            that can be used to perform various error checks and to raise
            exceptions while also automatically logging appropriate message on
            `log_chan`.
    """
    return _error_handlers.setdefault(log_chan.name, ErrorHandler(log_chan))


class ErrorHandler:
    """An error handler that provides reusable error checking methods and also handles logging
    error messages automatically.  Calling an error handler directly is equivalent to calling
    the `.log_raise` method.
    """

    def __init__(self, log_chan):
        """Create a new error handler that provides reusable error checking and automatic logging.

        Args:
            log_chan: alog channel
                The logging channel that this error handle will use for logging.
        """
        self.log_chan = log_chan

    def _handle_exception_messages(self, log_code, exception):
        """Handle number of exception log messages to avoid overflows"""
        # increment the log message counter attribute or add it if not present
        if hasattr(exception, "_caikit_core_nexception_log_messages"):
            exception._caikit_core_nexception_log_messages += 1
        else:
            exception._caikit_core_nexception_log_messages = 0

        caikit_config = get_config()

        # if less than max log messages, then omit a log message
        if (
            exception._caikit_core_nexception_log_messages
            < caikit_config.max_exception_log_messages
        ):
            self.log_chan.error(
                log_code, "exception raised: {}".format(repr(exception))
            )

        # if at the limit omit one message stating that we will no longer log
        elif (
            exception._caikit_core_nexception_log_messages
            == caikit_config.max_exception_log_messages
        ):
            self.log_chan.error(
                log_code,
                "reached MAX_EXCEPTION_LOG_MESSAGES of `{}`, will no log exception `{}`".format(
                    caikit_config.max_exception_log_messages, repr(exception)
                ),
            )

    def log_raise(self, log_code, exception, root_exception=None):
        """Log an exception with a log code and then re-raise it.  Using this instead of simply
        using the `raise` keyword with your exceptions will ensure that log message is emitted on
        the `error` level for the log channel associated with this handler.  This is invaluable for
        debugging in environments where stack traces are not available.

        Args:
            log_code (str): A log code with format `<COR12345678E>` where `COR`
                is a short code for the library (for `caikit_core` in this
                example) and where `12345678` is a unique eight-digit identifier
                (example generation in `scripts/cor_log_code`) and where `E` is
                an error level short-code, one of `{'fatal': 'F', 'error': 'E',
                'warning': 'W', 'info': 'I', 'trace': 'T', 'debug': 'D'}`.
            exception (Exception): A python exception object or an instance of
                any subclass of `Exception`.
            root_exception: Exception (Optional)
                A python exception object or an instance of any subclass of `Exception` which is
                considered a 'root' exception, wrapped by the preceding exception. Done in cases
                where we'd like to wrap a base exception with a custom exception, preserving the
                original's stack-trace
        Notes:
            The error handler will track the number log messages emitted for a given exception and
            stop logging after `MAX_EXCEPTION_LOG_MESSAGES` have been logged for a single instance
            of an exception.  This is to prevent pathological logging, e.g., repeatedly calling
            `log_raise` in a recursive function.
        """
        self._handle_exception_messages(log_code, exception)

        if root_exception:
            self._handle_exception_messages(log_code, root_exception)
            # reraise the exception chained with root_exception
            raise exception from root_exception

        # reraise the exception
        raise exception

    # calling an error handler is equivalent to calling the `.log_raise` method
    __call__ = log_raise

    def type_check(self, log_code, *types, allow_none=False, **variables):
        """Check for acceptable types for a given object.  If the type check fails, a log message
        will be emitted at the error level on the log channel associated with this handler and a
        `TypeError` exception will be raised with an appropriate message.  This check should be used
        to check for appropriate variable types.  For example, to verify that an argument passed to
        a function that expects a string is actually an instance of a string.

        Args:
            log_code (str): A log code with format `<COR90063501E>` where `COR`
                is a short code for the library (for `caikit_core` in this
                example) and where `90063501` is a unique eight-digit identifier
                (example generation in `scripts/cor_log_code`) and where `E` is
                an error level short-code, one of `{'fatal': 'F', 'error': 'E',
                'warning': 'W', 'info': 'I', 'trace': 'T', 'debug': 'D'}`.
            *types (type or None): Variadic arguments containing all acceptable
                types for `variables`.  If any values of `variable` are not any
                of `*types` then a log message will be emitted and a `TypeError`
                will be raised.  Multiple types may be specified as separate
                arguments. If no types are specified, then a `RuntimeError` will
                be raised.
            allow_none (bool): If `True` then the values of `variables` are
                allowed to take on the value of `None` without causing the type
                check to fail.  If `False` (default) then `None` values will
                cause the type check to fail.
            **variables (object): Variadic keyword arguments to be examined for
                acceptable type.  The name of the variable is used in log and
                error messages while its value is actually check against
                `types`.  Multiple keyword variables may be specified.  If no
                variables are specified, then a `RuntimeError` will be raised.

        Examples:
            # this will raise a `TypeError` because `foo` is not `None` or a `list` or `tuple`
            > error.type_check('<COR99962332E>', None, list, tuple, foo='hello world')

            # this type check verifies that `foo` and `bar` are both strings
            > error.type_check('<COR03761101E>', str, foo=foo, bar=bar)
        """
        if not get_config().enable_error_checks:
            return

        if not types:
            self(log_code, RuntimeError("invalid type check: no types specified"))

        if not variables:
            self(log_code, RuntimeError("invalid type check: no variables specified"))

        for name, variable in variables.items():
            if allow_none and variable is None:
                continue

            # check if variable is an instance of one of `types`
            if not isinstance(variable, types):
                type_name = type(variable).__name__
                valid_type_names = tuple(typ.__name__ for typ in types)
                if allow_none:
                    valid_type_names += (type(None).__name__,)

                # create, log and raise an appropriate exception
                self(
                    log_code,
                    TypeError(
                        "type check failed: variable `{}` has type `{}` not in `{}`".format(
                            name, type_name, valid_type_names
                        )
                    ),
                )

    def type_check_all(self, log_code, *types, allow_none=False, **variables):
        """This type check is similar to `.type_check` except that it verifies that each variable
        in `**variables` is either a `list` or a `tuple` and then checks that *all* of the items
        they contain are instances of a type in `*types`.  If `allow_none` is set to `True`, then
        the variable is allowed to be `None`, but the items in the `list` or `tuple` are not.

        Examples:
            # this type check will verify that foo is a `list` or `tuple` containing only `int`s
            > foo = (1, 2, 3)
            > error.type_check('<COR50993928E>', int, foo='hello world')

            # this type check allows `foo` to be `None`
            > error.type_check('<COR79540602E>', None, foo=None)

            # this type check fails because `foo` contains `None`
            > error.type_check('<COR87797257E>', None, int, foo=(1, 2, None, 3, 4))

            # this type check fails because `bar` contains a `str`
            # but not for any other reason
            > foo = [1, 2, 3]
            > bar = [4, 5, 'x']
            > baz = None
            > error.type_check('<COR40818868E>', None, int, foo=foo, bar=bar, baz=None)
        """
        if not get_config().enable_error_checks:
            return

        if not types:
            self(log_code, RuntimeError("invalid type check: no types specified"))

        if not variables:
            self(log_code, RuntimeError("invalid type check: no variables specified"))

        top_level_types = (Iterable,)
        invalid_types = (
            str,
            GeneratorType,
        )  # top level types that will fail the type check

        for name, variable in variables.items():
            if allow_none and variable is None:
                continue

            # log and raise if variable is not an Iterable
            if not isinstance(variable, top_level_types) or isinstance(
                variable, invalid_types
            ):
                type_name = type(variable).__name__
                valid_type_names = tuple(typ.__name__ for typ in top_level_types)
                if allow_none:
                    valid_type_names += (type(None).__name__,)

                self(
                    log_code,
                    TypeError(
                        "type check failed: variable `{}` has type `{}` not in `{}`".format(
                            name, type_name, valid_type_names
                        )
                    ),
                )

            # log and raise if any item is not in list of valid types
            for item in variable:
                if not isinstance(item, types):
                    type_name = type(item).__name__
                    valid_type_names = tuple(typ.__name__ for typ in types)

                    self(
                        log_code,
                        TypeError(
                            "type check failed: element of `{}` has type `{}` not in `{}`".format(
                                name, type_name, valid_type_names
                            )
                        ),
                    )

    def subclass_check(
        self, log_code: str, child_class: Any, *parent_classes, allow_none: bool = False
    ):
        """Check that the given child classes are valid types and that they
        derive from the given set of parent classes [issubclass(x, (y, z))]. If
        the subclass check fails, a log message will be emitted at the error
        level on the log channel associated with this handler and a `TypeError`
        exception will be raised with an appropriate message. This check should
        be used to check that a given class meets the interface of a parent
        class. For example, to verify that a class handle is a valid ModuleBase
        subclass.

        Args:
            log_code (str): A log code with format `<COR90063501E>` where `COR`
                is a short code for the library (for `caikit_core` in this
                example) and where `90063501` is a unique eight-digit identifier
                and where `E` is an error level short-code, one of `{'fatal':
                'F', 'error': 'E', 'warning': 'W', 'info': 'I', 'trace': 'T',
                'debug': 'D'}`.
            child_class (Any): The class to be examined for acceptable class
                inheritance.
            *parent_classes (type): Variadic arguments containing all acceptable
                parent types for `child_classes`.  If any values of
                `child_classes` are not a valid type derived from one of
                `*parent_classes` then a log message will be emitted and a
                `TypeError` will be raised. Multiple parent_classes may be
                specified as separate arguments. If no parent_classes are
                specified, then a `RuntimeError` will be raised.
            allow_none (bool): If `True` then the values of `child_classes` are
                allowed to take on the value of `None` without causing the
                subclass check to fail.  If `False` (default) then `None` values
                will cause the subclass check to fail.

        Examples:
            # this will raise a `TypeError` because `Foo` is not `None` or
            # derived from Bar
            > class Bar: pass
            > class Foo: pass
            > error.subclass_check('<COR99962332E>', Bar, Foo=Foo)

            # this type check verifies that `foo` and `bar` are both strings
            > error.type_check('<COR03761101E>', str, foo=foo, bar=bar)
        """
        if not get_config().enable_error_checks:
            return

        if allow_none and child_class is None:
            return

        if not parent_classes:
            self(
                log_code,
                RuntimeError("invalid subclass check: no parent_classes given"),
            )

        if not isinstance(child_class, type) or not issubclass(
            child_class, parent_classes
        ):
            self(
                log_code,
                TypeError(
                    "subclass check failed: {} is not a subclass of {}".format(
                        child_class,
                        parent_classes,
                    )
                ),
            )

    def value_check(self, log_code, condition, *args):
        """Check for acceptable values for a given object.  If this check fails, a log message will
        be emitted at the error level on the log channel associated with this handler and a
        `ValueError` exception will be raised with an appropriate message.  This check should be
        used for checking for appropriate values for variable instances.  For example, to check that
        a numerical value has an appropriate range.

        Args:
            log_code (str): A log code with format `<COR55705215E>` where `COR`
                is a short code for the library (for `caikit_core` in this
                example) and where `55705215` is a unique eight-digit identifier
                (example generation in `scripts/cor_log_code`) and where `E` is
                an error level short-code, one of `{'fatal': 'F', 'error': 'E',
                'warning': 'W', 'info': 'I', 'trace': 'T', 'debug': 'D'}`.
            condition (bool): A boolean value that should describe if this check
                passes `True` or fails `False`. Upon calling this function, this
                is typically provided as an expression, e.g., `0 < variable <
                1`.
            *args: A variable set of arguments describing the value check that failed. If no
                args are provided then an empty msg string is assumed and no additional
                information will be provided, otherwise the first argument will be treated as 'msg'
                argument. Note that string interpolation can be lazily performed on `msg` using `{}`
                format syntax by passing additional arguments.  This is the preferred method for
                performing string interpolation on `msg` so that it is only done if an error
                condition is encountered.

        """
        if not get_config().enable_error_checks:
            return

        if not condition:
            interpolated_msg = (
                ""
                if not args
                else (args[0] if len(args) == 1 else args[0].format(*args[1:]))
            )

            self(
                log_code, ValueError("value check failed: {}".format(interpolated_msg))
            )

    def file_check(self, log_code, *file_paths):
        """Check to see if one or more file paths exist and are regular files.  If any do not exist
        or are not files, then a log message will be emitted on the log channel associated with this
        error handler and a `FileNotFoundError` will be raised with an appropriate error message.

        Args:
            log_code (str): A log code with format `<COR73692990E>` where `COR`
                is a short code for the library (for `caikit_core` in this
                example) and where `55705215` is a unique eight-digit identifier
                (example generation in `scripts/cor_log_code`) and where `E` is
                an error level short-code, one of `{'fatal': 'F', 'error': 'E',
                'warning': 'W', 'info': 'I', 'trace': 'T', 'debug': 'D'}`.
            *file_paths (str): Variadic argument containing strings specifying
                the file paths to check.  If any of these file paths does not
                exist or is not a regular file, then a log message will be
                emitted and a `FileNotFoundError` will be raised.
        """
        if not get_config().enable_error_checks:
            return

        for file_path in file_paths:
            if not os.path.exists(file_path):
                self(
                    log_code,
                    FileNotFoundError(
                        "File path `{}` does not exist".format(file_path)
                    ),
                )

            if not os.path.isfile(file_path):
                self(
                    log_code,
                    FileNotFoundError("Path `{}` is not a file".format(file_path)),
                )

    def dir_check(self, log_code, *dir_paths):
        """Check to see if one or more directory paths exist and are, in fact, directories.  If any
        do not exist then a `FileNotFoundError` will be raised and if they are not directories then
        a `NotADirectoryError` will be raised.  In either case, a log message will be emitted on the
        log channel associated with this error handler.

        Args:
            log_code (str): A log code with format `<COR63462828E>` where `COR`
                is a short code for the library (for `caikit_core` in this
                example) and where `55705215` is a unique eight-digit identifier
                (example generation in `scripts/cor_log_code`) and where `E` is
                an error level short-code, one of `{'fatal': 'F', 'error': 'E',
                'warning': 'W', 'info': 'I', 'trace': 'T', 'debug': 'D'}`.
            *dir_paths (str): Variadic argument containing strings specifying
                the directory paths to check.  If any of these file paths does
                not exist or is not a regular file, then a log message will be
                emitted and a `FileNotFoundError` or `NotADirectoryError` will
                raised.
        """
        if not get_config().enable_error_checks:
            return

        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                self(
                    log_code,
                    FileNotFoundError(
                        "Directory path `{}` does not exist".format(dir_path)
                    ),
                )

            if not os.path.isdir(dir_path):
                self(
                    log_code,
                    NotADirectoryError("Path `{}` is not a directory".format(dir_path)),
                )
