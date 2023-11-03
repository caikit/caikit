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
"""
An ExceptionPickler deals with deconstructing any `Exception` type into picklable
parts so that it can be passed across a subprocess boundary without either failing
to un-pickle or losing important context.

The python BaseException class implements its own __reduce__ method so that all
subclasses support pickling, but it has some intentional drawbacks:
1. It can't know about kwarg arguments to __init__, so it only supports subclasses
that have *arg initializers. However, many custom Exception types contain keyword
arguments in their __init__ methods which cause an unpickling failure.
2. It does not pickle the __cause__ or __context__ of an exception, presumably
because it can't make any guarantees about the picklability of those objects. This
leads to one-line tracebacks on the unpickled exception, because there is no context
to generate a useful stack trace from.
"""
import pickle
import re


class PickleFailureFallbackException(Exception):
    """Exception type used to replace exceptions that just cannot be pickled no matter
    how hard we try."""


class ExceptionPickler:

    # Matches the specific TypeError that raises when exception classes allow init kwargs but do not handle them in __reduce__
    _type_error_expression = re.compile(r".*__init__\(\) missing \d required positional argument")
    # Matches the names of the positional arguments that are missing, from the TypeError's string
    _arg_match_expression = re.compile(r".*?'(.+?)'+")


    def __init__(self, exception: BaseException):
        """
        Args:
            exception: The exception that will be safely pickled within this container
        """
        # TODO: value check not none
        self.exception = exception

    def get(self) -> BaseException:
        """Returns the exception, reconstructed after pickling as best as possible"""
        return self.exception

    def __setstate__(self, state_dict):
        """Reconstructs the exception out of the state_dict that is returned by __getstate__"""
        if "exception" in state_dict:
            self.exception = state_dict["exception"]
        else:
            initializer = state_dict["initializer"]
            self.exception = initializer(*state_dict["args"], **state_dict["kwargs"])

        if state_dict["cause"]:
            self.exception.__cause__ = state_dict["cause"].get()

        if state_dict["context"]:
            self.exception.__context__ = state_dict["context"].get()

    def __getstate__(self) -> dict:
        """Package up the exception's details into a dict, taking care to:
        - include the __cause__ and __context__, which are not serialized by default
            - Recursively wrap _those_ in PicklingExceptionWrappers
        - Check that this exception _can_ be pickled, and try to handle common problems with __reduce__
        """
        state_dict = {
            "exception": self.exception,
            "cause": ExceptionPickler(self.exception.__cause__) if self.exception.__cause__ else None,
            "context": ExceptionPickler(self.exception.__context__) if self.exception.__context__ else None
        }

        # ty/catch pickle errors
        try:
            pickle.loads(pickle.dumps(self.exception))
        except TypeError as type_error:
            if self._type_error_expression.match(str(type_error)):
                try:
                    keywords = self._arg_match_expression.findall(str(type_error))

                    # First grab the positional arguments. This should be provided by BaseException
                    args = self.exception.args
                    # Then look for each kwarg
                    kwargs = {}
                    for kwarg in keywords:
                        # Try to fetch the attribute
                        if hasattr(self.exception, kwarg):
                            arg = getattr(self.exception, kwarg)
                            kwargs[kwarg] = arg
                        elif hasattr(self.exception, f"_{kwarg}"):
                            arg = getattr(self.exception, f"_{kwarg}")
                            kwargs[kwarg] = arg
                        else:
                            raise ValueError(f"{self.exception} has no attributes matching kwarg name {kwarg}")

                    state_dict.pop("exception")
                    state_dict["initializer"] = type(self.exception)
                    state_dict["args"] = args
                    state_dict["kwargs"] = kwargs

                    # check that we can re-build this exception
                    _ = state_dict["initializer"](*state_dict["args"], **state_dict["kwargs"])
                except Exception as e:
                    state_dict["exception"] = PickleFailureFallbackException(str(self.exception))
            else:
                state_dict["exception"] = PickleFailureFallbackException(str(self.exception))




        # if TypeError =.__init__() missing %d required positional arguments: 'attr1' and 'attr2'...
        # instead try to store the class to initialize along with `exception.attr1`, `exception.attr2` etc
        # if that fails or other error then...
        # Init a PickleFallbackException with str(exception)

        return state_dict