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
This module holds the module_type decorator which is used to declare subtypes
of ModuleBase (e.g. Block)
"""

# Standard
from typing import Dict, Optional, Type, Union
import collections

# Third Party
import semver

# First Party
import alog

# Local
from .. import core
from . import data_model as dm
from .module import _MODULE_TYPES, MODULE_BACKEND_REGISTRY, MODULE_REGISTRY, ModuleBase
from .module_backends import backend_types
from .signature_parsing import CaikitMethodSignature
from .task import TaskBase
from .toolkit.errors import error_handler

log = alog.use_channel("MODTYP")
error = error_handler.get(log)


# This is used by individual caikit.module implementations, like a block
# to define what backend type models they can support at load time.
SUPPORTED_LOAD_BACKENDS_VAR_NAME = "SUPPORTED_LOAD_BACKENDS"


def module_type(module_type_name):
    """This encapsulates the logic of creating a derived module subtype
    (e.g. block). It is intended to decorate a class which inherits from
    ModuleBase. The wrapped class is augmented in the following ways:

    * A new class-attribute is defined named `module_type` which is itself a
        decorator that concrete implementations of this module type can use to
        bind a module id, description, and version (e.g. @BlockBase.block).
    * A new class attribute is added as a global registry that the above
        decorator will use to store all registered concrete implementations of
        the module type

    These above class attributes can be further hoisted to free attributes in
    the python module where the module type is defined (e.g. @block).
    """
    assert (
        module_type_name == module_type_name.lower()
    ), "All module types must be lowercase"
    module_type_name_upper = module_type_name.upper()
    _MODULE_TYPES.append(module_type_name_upper)

    def module_type_decorator(cls):
        # Perform top-level imports here for places that the the new module type
        # will need to be added. This is done to avoid circular dependencies
        # since these top-level imports are only needed where the decorator is
        # used.
        # Local
        # pylint C0415: Import outside toplevel (import-outside-toplevel)
        # pylint: disable=import-outside-toplevel
        from caikit.core import model_manager

        # Add the registry dict and decorate the top-level module
        registry_name = f"{module_type_name_upper}_REGISTRY"
        impl_registry = {}
        error.value_check(
            "<COR68254470E>",
            not hasattr(core, registry_name),
            "Cannot re-declare module type {}",
            module_type_name,
        )
        setattr(core, registry_name, impl_registry)
        setattr(model_manager, registry_name, impl_registry)
        setattr(cls, "REGISTRY", impl_registry)

        # Define the module implementation decorator
        # pylint: disable=redefined-builtin,pointless-statement
        def _module_impl_decorator(
            id=None,
            name=None,
            version=None,
            task: Type[TaskBase] = None,
            backend_type=backend_types.LOCAL,
            base_module: Union[str, Type[ModuleBase]] = None,
            backend_config_override: Optional[Dict] = None,
        ):
            f"""Apply this decorator to any class that should be treated as a {module_type_name}
             (i.e., extends`{cls.__name__}) and registered with caikit.core so that the library
             "knows" the class is a {module_type_name} and is capable of loading instances of the
             {module_type_name}.

            Args:
                id:  str
                    A UUID to use when registering this {module_type_name} with caikit.core
                    Not required if based on another caikit module using `base_module`
                name:  str
                    A human-readable name for the {module_type_name}
                    Not required if based on another caikit module using `base_module`
                version:  str
                    A SemVer for the {module_type_name}
                    Not required if based on another caikit module using `base_module`
                task:  Type[TaskBase]
                    An ML task class that this module is an implementation for
                    Not required if based on another caikit module using `base_module`
                backend_type: backend_type
                    Associated backend type for the module.
                    Default: `LOCAL`
                base_module: str | ModuleBase
                    If this module is based on a different caikit module, provide name
                    of the base module.
                    Default: None
                backend_config_override: Dict
                    Dictionary containing configuration required for the specific backend.
                    Default: None

            Returns:
                A decorated version of the class to which it was applied, after registering the
                class as a valid {module_type_name} with caikit.core
            """
            base_module_class = None
            # Flag to store if the current module is a backend implementation
            # of an existing module or not
            backend_module_impl = False

            # No mutable default
            backend_config_override = backend_config_override or {}

            if any([id is None, version is None or name is None]):
                error.type_check(
                    "<COR87944440E>",
                    str,
                    type(ModuleBase),
                    allow_none=False,
                    base_module=base_module,
                )
                error.type_check(
                    "<COR60584425E>",
                    dict,
                    allow_none=True,
                    backend_config_override=backend_config_override,
                )

                # If the base_module is a string, assume that it is the module_id of the
                # base module
                if isinstance(base_module, str):
                    module_id = base_module
                    error.value_check(
                        "<COR09479833E>",
                        module_id in MODULE_REGISTRY,
                        "Unknown base module id: {}",
                        module_id,
                    )
                    base_module_class = MODULE_REGISTRY[module_id]

                # If base_module is a type, validate that it derives from ModuleBase and
                # use its MODULE_ID
                elif isinstance(base_module, type):
                    if not issubclass(base_module, ModuleBase):
                        error(
                            "<COR20161747E>",
                            f"base_module [{base_module}] does not derive from ModuleBase",
                        )

                    base_module_class = base_module

                # TODO: Add support for inheritance of backend implementation
                # i.e if a module inherits from base_module

                id = base_module_class.MODULE_ID
                version = base_module_class.MODULE_VERSION
                name = base_module_class.MODULE_NAME
                task = base_module_class.TASK_CLASS
                backend_module_impl = True

            error.type_check("<COR54118928E>", str, id=id, name=name, version=version)
            error.subclass_check("<COR90789722E>", task, TaskBase, allow_none=True)

            semver.VersionInfo.parse(version)  # Make sure this is a valid SemVer

            def decorator(cls_):
                # Verify this is a valid module type (inherits from the wrapped base class)
                if not issubclass(cls_, cls):
                    error(
                        "<COR32401861E>",
                        TypeError(
                            f"`{cls_.__name__}` does not extend `{cls.__name__}`",
                        ),
                    )

                # Add attributes to the implementation class
                setattr(cls_, f"{module_type_name_upper}_ID", id)
                cls_.MODULE_ID = id  # Module ID == Module Type ID
                setattr(cls_, f"{module_type_name_upper}_NAME", name)
                cls_.MODULE_NAME = name  # Module Name == Module Type Name
                setattr(cls_, f"{module_type_name_upper}_VERSION", version)
                cls_.MODULE_VERSION = version  # Module Version == Module Type Version
                classname = f"{cls_.__module__}.{cls_.__qualname__}"
                setattr(cls_, f"{module_type_name_upper}_CLASS", classname)
                cls_.MODULE_CLASS = classname
                cls_.PRODUCER_ID = dm.ProducerId(cls_.MODULE_NAME, cls_.MODULE_VERSION)

                # Tasks: check to see if a super-class has one as well and that they match:
                tasks = {
                    class_.TASK_CLASS
                    for class_ in cls_.mro()
                    if hasattr(class_, "TASK_CLASS")
                }
                if len(tasks) > 1:
                    error(
                        "<COR17197749E>",
                        TypeError(
                            f"Class {cls_} has multiple task definitions in class hierarchy"
                        ),
                    )
                if tasks:
                    parent_task = tasks.pop()
                    if task and task != parent_task:
                        error(
                            "<COR44943734E>",
                            TypeError(
                                f"Class {cls_} has task {task} but superclass has task "
                                f"{parent_task}"
                            ),
                        )
                    cls_.TASK_CLASS = parent_task
                else:
                    cls_.TASK_CLASS = task

                # Parse the `train` and `run` signatures
                cls_.RUN_SIGNATURE = CaikitMethodSignature(cls_, "run")
                cls_.TRAIN_SIGNATURE = CaikitMethodSignature(cls_, "train")

                # If the module has a task, validate it:
                if cls_.TASK_CLASS:
                    cls_.TASK_CLASS.validate_run_signature(cls_.RUN_SIGNATURE)

                # Set module type as attribute of the class
                # pylint: disable=global-variable-not-assigned
                cls_.MODULE_TYPE = module_type_name_upper

                # If no backend support described in the class, add current backend
                # as the only backend that can load models trained by this module
                cls_.SUPPORTED_LOAD_BACKENDS = getattr(
                    cls_, SUPPORTED_LOAD_BACKENDS_VAR_NAME, [backend_type]
                )

                # Set its own backend_type as an attribute
                setattr(cls_, "BACKEND_TYPE", backend_type)

                # Verify UUID and add this block to the module and block registries
                global MODULE_REGISTRY
                current_class = MODULE_REGISTRY.get(cls_.MODULE_ID)
                if not backend_module_impl:
                    if current_class is not None:
                        error(
                            "<COR30607646E>",
                            RuntimeError(
                                "MODULE_ID `{}` conflicts for classes `{}` and `{}`".format(
                                    cls_.MODULE_ID,
                                    cls_.__name__,
                                    MODULE_REGISTRY[cls_.MODULE_ID].__name__,
                                )
                            ),
                        )
                    MODULE_REGISTRY[cls_.MODULE_ID] = cls_
                    impl_registry[cls_.MODULE_ID] = cls_

                # Register backend
                _register_module_implementation(
                    cls_, backend_type, cls_.MODULE_ID, backend_config_override
                )

                return cls_

            return decorator

        # Add this decorator to the wrapped class
        setattr(cls, module_type_name, _module_impl_decorator)
        return cls

    return module_type_decorator


## Implementation Details ######################################################


def _register_module_implementation(
    implementation_class: type,
    backend_type: str,
    module_id: str,
    backend_config_override: Dict = None,
):
    """This function will register the mapping for the given module_id and
    backend_type to the implementation class

    Args:
        implementation_class:  type
            The class that is used to implement this backend type for the given
            module_id
        backend_type:  str
            Value from MODULE_BACKEND_TYPES that indicates the backend
            that this class implements
        module_id:  str
            The module_id from the caikit.core module registry that this class
            overloads
        backend_config_override: Dict
            Dictionary containing essential overrides for the backend config.
            This will get stored with the implementation_class class name and will automatically
            get picked up and merged with other such configs for a specific backend

    """

    backend_config_override = backend_config_override or {}

    log.debug(
        "Registering backend [%s] implementation for module [%s]",
        backend_type,
        module_id,
    )

    error.value_check(
        "<COR86780140E>",
        backend_type in backend_types.MODULE_BACKEND_TYPES,
        "Cannot override implementation of {} for unkonwn backend type {}",
        module_id,
        backend_type,
    )

    core_class = MODULE_REGISTRY.get(module_id)
    if core_class is None:
        # TODO! Inject a dummy entry that will raise on usage
        pass  # pragma: no cover

    # Do the registration!
    module_type_mapping = MODULE_BACKEND_REGISTRY.setdefault(module_id, {})

    # Make sure this is not an overwrite of an existing registration
    existing_type = module_type_mapping.get(backend_type)
    assert (
        existing_type is None or existing_type is implementation_class
    ), f"Registration conflict! ({module_id}, {backend_type}) already registered as {existing_type}"

    BackendConfig = collections.namedtuple(
        "BackendConfig", "impl_class backend_config_override"
    )

    module_type_mapping[backend_type] = BackendConfig(
        impl_class=implementation_class, backend_config_override=backend_config_override
    )
