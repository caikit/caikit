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


"""This module contains an implementation of a metaclass that hijacks all
${class}.load(...) invocations.

For a good read on metaclasses, see https://realpython.com/python-metaclasses/

Our goal is to have every instance of a `caikit.core.ModuleBase` automatically populated with
metadata when it is constructed, without requiring anything of the module authors.
The source of this metadata will be the `config.yml` file that resides within the module directory
to be loaded. Since we require the path to that file in order to read the model's metadata, we
cannot simply define a base constructor. Instead, we patch over the module's .load() function,
which is guaranteed to be called with a path to a serialized module.

Additionally, the naive solution of manually patching .load functions does not work when
inheritance is involved. For example::

    import caikit.core

    class ParentModule(caikit.core.ModuleBase):

        @classmethod
        def load(cls, module_path):
            return cls()

    class ChildModule(ParentModule):

        @classmethod
        def load(cls, module_path):
            return super().load()

    # This is fine!
    assert isinstance(ChildModule.load(), ChildModule)

    def injector(load_fn):
        def injected_load(*args):
            module = load_fn(*args)
            module.metadata = {"stuff"}
            return module
        return classmethod(injected_load)

    for clz in (ParentModule, ChildModule):
        # But this line binds the new load function directly to each class :(
        clz.load = injector(clz.load)

    # And this will now raise since ParentModule is returned
    assert isinstance(ChildModule.load(), ChildModule)


Instead of binding new metadata-injecting load functions directly to a class at import time, we
need to bind the new load function at contruction time, when the class hierarchy with inheritance
is known.
"""

# Standard
import abc
import functools

# First Party
import alog

# Local
from .config import ModuleConfig
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("METADATA_INJECT")
error = error_handler.get(log)


class _ModuleBaseMeta(abc.ABCMeta):
    """This is the metaclass used by `caikit.core.ModuleBase`.

    This metaclass populates the `metadata` property of any module that is created by invoking
    a `load` classmethod on a derived class.
    """

    # pylint: disable=arguments-differ
    def __new__(mcs, name, bases, attrs):
        real_load = attrs.get("load")
        if real_load is not None:
            log.debug3("Wrapping a load function on class %s", mcs)

            @alog.logged_function(log.trace)
            def metadata_injecting_load(clz, *args, **kwargs):
                """This function is the replacement for the module's original `.load`"""
                path = None
                module_config = None

                # Many load functions rename what the `path` argument is.
                # Usually, it's just the first positional argument.
                # But, `load` may be called with kwargs-only, like:
                # MyModelClass.load(model_path="/some/path")
                log.debug3(
                    "Attempting to find model path from load args: [%s] [%s]",
                    args,
                    kwargs,
                )
                if len(args) > 0:
                    log.debug3(
                        "Using first positional argument to load as model path: %s",
                        args[0],
                    )
                    path = args[0]
                else:
                    for kw, arg in kwargs.items():
                        if "path" in kw:
                            log.debug3(
                                "Using named keyword argument to load as model path: [%s: %s]",
                                kw,
                                arg,
                            )
                            path = arg
                            break
                try:
                    if path:
                        module_config = ModuleConfig.load(path)
                except FileNotFoundError:
                    log.error(
                        "Could not load module metadata while loading with %s %s",
                        args,
                        kwargs,
                    )

                # Call the original .load function to load the module
                module = real_load.__func__(clz, *args, **kwargs)

                # defer any "is this really a module" logic until after the load call
                if hasattr(module, "metadata") and module_config:
                    module.metadata.update(module_config)

                return module

            # Wrap the load function so that the final method appears the same
            # as the original
            metadata_injecting_load = functools.wraps(real_load.__func__)(
                metadata_injecting_load
            )
            attrs["load"] = classmethod(metadata_injecting_load)

        return super().__new__(mcs, name, bases, attrs)

    def __setattr__(cls, name, val):
        """Overwrite __setattr__ to warn on any dynamic updates to the load function.
        We'd rather not lose all the work we did to wrap `.load` with metadata injection
        in the constructor!
        """
        if name == "load":
            # NB: warn instead of throw because some libraries will mock out .load during
            # unit testing where it's easier than trying to build a quick-loading dummy model
            log.warning("Overwriting load on a module will break metadata persistence!")
        return super().__setattr__(name, val)
