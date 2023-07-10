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
from enum import Enum
import inspect

# First Party
# First party
import alog

# Local
from .errors import error_handler

log = alog.use_channel("WIPDC")
error = error_handler.get(log)

################ Constants ##########################################

message_format = "{} is still in the {} phase and subject to change!"

_ENABLE_DECORATOR = True


class WipCategory(Enum):
    WIP = 1
    BETA = 2


class Action(Enum):
    ERROR = 1
    WARNING = 2


################ Implementation #####################################


def disable_wip():
    """Utility function to disable decorator functionality.
    Mainly designed for testing
    """
    # pylint: disable=global-statement
    global _ENABLE_DECORATOR
    _ENABLE_DECORATOR = False


def enable_wip():
    """Utility function to enable decorator functionality.
    Mainly designed for testing
    """
    # pylint: disable=global-statement
    global _ENABLE_DECORATOR
    _ENABLE_DECORATOR = True


class TempDisableWIP:
    """Temporarily disable wip decorator for a particular block of code

    NOTE: There is a potential race condition possible here in cases where
    other code using wip decorator gets called at the same time this context
    based disabling functionality is invoked. If this happens, the decorator will
    get disabled for all the functions invoking at the same time. This is
    because we are using the global disable / enable functions for this class.
    """

    ## Notes to fix:
    # As per current design of the decorator the enable / disable can only happen
    # globally at the invocation time. May be there is a way to overcome this by
    # changing the _get_message function dynamically.

    def __enter__(self):
        disable_wip()

    def __exit__(self, *args):
        enable_wip()


def work_in_progress(*args, **kwargs):
    """Decorator that can be used to mark a function
    or a class as "work in progress". It will result in a warning being emitted
    when the function / class is used.

    Args:
        category (WipCategory): Enum specifying what category of message you
            want to throw
        action (Action): Enum specifying what type of action you want to take.
            Example: ERROR or WARNING

    Example Usage:

    ### Decorating class

    1. No configuration:
        @work_in_progress
        class Foo:
            pass

    2. Action and category configuration:
        @work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
        class Foo:
            pass

    ### Decorating Function:

    1. No configuration:
        @work_in_progress
        def foo(*args, **kwargs):
            pass

    2. Action and category configuration:
        @work_in_progress(action=Action.WARNING, category=WipCategory.BETA)
         def foo(*args, **kwargs):
            pass

    ### Sample message:

    foo is still in the BETA phase and subject to change!

    """
    if args:
        wrapped_obj = args[0]
    else:
        wrapped_obj = None

    # Set defaults
    category = kwargs.get("category", WipCategory.WIP)
    action = kwargs.get("action", Action.WARNING)

    # Type checks
    error.type_check("<TRU92076783E>", WipCategory, category=category)
    error.type_check("<TRU87572165E>", Action, action=action)

    if inspect.isclass(wrapped_obj) or inspect.isfunction(wrapped_obj):
        # TODO: if a class, add this decorator to all the functions of this class
        return _decorator_handler(wrapped_obj, category, action)

    if len(kwargs) > 0:

        def decorator(wrapped_obj):
            return _decorator_handler(wrapped_obj, category, action)

        return decorator

    raise ValueError(
        "Invalid usage of wip decorator. {} argument not supported!".format(
            type(wrapped_obj)
        )
    )


def _decorator_handler(wrapped_obj, category, action):
    """Utility function to cover common decorator handling
    logic.
    Args:
        wrapped_obj (Callable): Class or function to be decorated
        category (Enum(WipCategory)): Enum specifying the category of the
            message
        Action (Enum(Action)): Enum specifying the action to be taken with the
            decorator
    Returns:
        function:
            Decorator function
    """

    if inspect.isclass(wrapped_obj):
        # Replace __new__ function of wrapped class
        # with wrapped_cls function that includes
        # warning message
        new_class = wrapped_obj.__new__

        def wrapped_cls(cls, *args, **kwargs):
            _get_message(wrapped_obj, category, action)

            # if class __new__ is empty
            if new_class is object.__new__:
                return new_class(cls)

            return new_class(cls, *args, **kwargs)

        wrapped_obj.__new__ = staticmethod(wrapped_cls)

        return wrapped_obj

    def wip_decorator(*args, **kwargs):
        # function specific handling
        _get_message(wrapped_obj, category, action)
        return wrapped_obj(*args, **kwargs)

    return wip_decorator


def _get_message(wrapped_obj, category, action):
    """Utility function to run action"""
    if _ENABLE_DECORATOR:
        message = message_format.format(wrapped_obj, category.name)
        if action == Action.ERROR:
            raise RuntimeError(message)
        if action == Action.WARNING:
            log.warning(message)
