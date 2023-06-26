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
import importlib
import sys

# First Party
import alog

# Local
from .errors import error_handler

log = alog.use_channel("COMPATIBILITY")
error = error_handler.get(log)


def unsupported_platforms(platforms):
    """This decorator can be used to raise NotImplementedError if it's run on a given platform.
    Otherwise it runs as normal. For guidance on different permissible values of platforms,
    see: https://docs.python.org/3/library/sys.html#sys.platform

    Args:
        platforms (List | Tuple | str): Platforms on which this capability is
            not implemented.
    """
    error.type_check("<COR48137162E>", list, tuple, str, platforms=platforms)
    if not isinstance(platforms, str):
        error.type_check_all("<COR92928316E>", str, platforms=platforms)

    def decorator(func):
        def wrapper(*args, **kwargs):
            if (
                isinstance(platforms, str) and sys.platform == platforms
            ) or sys.platform in platforms:
                error(
                    "<COR12233351E>",
                    NotImplementedError(
                        "Operation [{}] has not been implemented for platform [{}]".format(
                            func.__name__, sys.platform
                        )
                    ),
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_importable_on_platform(module, package, exc_type, platforms, platform_hint):
    """Some modules have trouble on specific operating systems. This function takes
    a module and package from which we want to import, and an exception type to look
    out for on a given problem. If the exception is raised, we warn the platform hint.

    NOTE: This function does set any properties on your package. You should use this
    function to check if something is importable, then you should import it. This will
    only run the package initialization one time, because this function will add your
    module into the loaded packages if it's successful, but doing things this way prevents
    your package from having a bunch of undesirable attributes and keeps it cleaner than other
    approaches. It also ensures that the import stack trace doesn't look abnormally handled if we
    raise on other platforms.

    Args:
        module (str): Module that we want to import.
        package (str): Package from which we want to import the module.
        exc_type (Exception): Exception that we want to warn a hint for if we
            hit it on import for a given platform.
        platforms (str): Platform on which we expect potential bad import from
            for this module.
        platform_hint (str): Hint to warn if we hit the exception type on the
            provided platform.
    """
    error.type_check(
        "<COR48442162E>",
        str,
        module=module,
        package=package,
        platform_hint=platform_hint,
    )
    error.type_check("<COR62245359E>", list, tuple, str, platforms=platforms)
    if not isinstance(platforms, str):
        error.type_check_all("<COR86935489E>", str, platforms=platforms)
    # If this is not a platform that needs special handling, just return True
    if (
        isinstance(platforms, str) and sys.platform != platforms
    ) or sys.platform not in platforms:
        return True
    # We are running on a platform that has potential issues for this module.
    # If we succeed on import, return True. If we throw the known exception type,
    # catch it and log the hint, then return False. This is non-fatal.
    try:
        log.debug(
            "Determining if module {} is importable on platform {}".format(
                module, platforms
            )
        )
        importlib.import_module(module, package)
        return True
    except exc_type as e:
        log.warning(
            "Module [{}] from package [{}] failed to import on platform [{}]"
            " with error [{}] and will not be available.\nHINT: {}".format(
                module, package, platforms, str(e), platform_hint
            )
        )
        return False
