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
This module holds the @module decorator which is used to decorate a
caikit.module
"""

# Standard
from typing import Dict, List, Optional, Type, Union
import collections

# Third Party
import semver

# First Party
import alog

# Local
from ..data_model import ProducerId
from ..exceptions import error_handler
from ..registries import module_backend_registry, module_backend_types, module_registry
from ..signature_parsing import CaikitMethodSignature
from ..task import TaskBase
from .base import ModuleBase
import caikit.core

log = alog.use_channel("MODULE_DEC")
error = error_handler.get(log)


# This is used by individual caikit.module implementations,
# to define what backend type models they can support at load time.
SUPPORTED_LOAD_BACKENDS_VAR_NAME = "SUPPORTED_LOAD_BACKENDS"


def module(
    id=None,  # pylint: disable=redefined-builtin
    name=None,
    version=None,
    task: Type[TaskBase] = None,
    tasks: Optional[List[Type[TaskBase]]] = None,
    backend_type="LOCAL",
    base_module: Union[str, Type[ModuleBase]] = None,
    backend_config_override: Optional[Dict] = None,
):
    """Apply this decorator to any class that should be treated as a caikit module
     (i.e., extends`{caikit.core.ModuleBase}) and registered with caikit.core so that the library
     "knows" the class is a caikit module and is capable of loading instances of the module.

    Args:
        id:  str
            A UUID to use when registering this module with caikit.core
            Not required if based on another caikit module using `base_module`
        name:  str
            A human-readable name for the module
            Not required if based on another caikit module using `base_module`
        version:  str
            A SemVer for the module
            Not required if based on another caikit module using `base_module`
        task:  Type[TaskBase]
            An ML task class that this module is an implementation for
            Not required if based on another caikit module using `base_module`,
            or if multiple tasks are specified using `tasks`.
        tasks: Optional[List[Type[TaskBase]]
            List of ML task classes that this module implements.
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
        class as a valid module with caikit.core
    """
    base_module_class = None
    # Flag to store if the current module is a backend implementation
    # of an existing module or not
    backend_module_impl = False

    # No mutable default
    backend_config_override = backend_config_override or {}

    if task and tasks:
        error(
            "<COR34125316E>",
            ValueError("Specify either task or tasks parameter, not both."),
        )
    if tasks:
        error.type_check(
            "<COR34125317E>",
            list,
            allow_none=True,
            tasks=tasks,
        )

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
                module_id in module_registry(),
                "Unknown base module id: {}",
                module_id,
            )
            base_module_class = module_registry()[module_id]

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
        tasks = base_module_class._TASK_CLASSES
        backend_module_impl = True

    if task is not None:
        tasks = [task]

    if tasks is None:
        tasks = []

    error.type_check("<COR54118928E>", str, id=id, name=name, version=version)

    for t in tasks:
        error.subclass_check("<COR90789722E>", t, TaskBase, allow_none=True)

    semver.VersionInfo.parse(version)  # Make sure this is a valid SemVer

    def decorator(cls_):
        # Verify this is a valid module type (inherits from the wrapped base class)
        if not issubclass(cls_, caikit.core.ModuleBase):
            error(
                "<COR32401861E>",
                TypeError(
                    f"`{cls_.__name__}` does not extend ModuleBase",
                ),
            )

        # Add attributes to the implementation class
        cls_.MODULE_ID = id  # Module ID == Module Type ID
        cls_.MODULE_NAME = name  # Module Name == Module Type Name
        cls_.MODULE_VERSION = version  # Module Version == Module Type Version
        classname = f"{cls_.__module__}.{cls_.__qualname__}"
        cls_.MODULE_CLASS = classname
        cls_.PRODUCER_ID = ProducerId(cls_.MODULE_NAME, cls_.MODULE_VERSION)

        # Parse the `train` and `run` signatures
        cls_.RUN_SIGNATURE = CaikitMethodSignature(cls_, "run")
        cls_.TRAIN_SIGNATURE = CaikitMethodSignature(cls_, "train")
        cls_._TASK_INFERENCE_SIGNATURES = {}

        # If the module has tasks, validate them:
        task_classes = tasks
        for t in task_classes:
            if not t.has_inference_method_decorators(module_class=cls_):
                # Hackity hack hack - make sure at least one flavor is supported
                validated = False
                validation_errs = []
                for input_streaming, output_streaming in [
                    [False, False],
                    [True, True],
                    [False, True],
                ]:
                    try:
                        t.validate_run_signature(
                            cls_.RUN_SIGNATURE, input_streaming, output_streaming
                        )
                        validated = True
                        cls_._TASK_INFERENCE_SIGNATURES.setdefault(t, []).append(
                            (input_streaming, output_streaming, cls_.RUN_SIGNATURE)
                        )
                        break
                    except (ValueError, TypeError) as e:
                        validation_errs.append(e)
                if not validated:
                    raise validation_errs[0]

            t.deferred_method_decoration(cls_)

        # Check to see if a super-class has any tasks.
        # These will have been validated by the superclass decorator already.
        tasks_in_hierarchy = []

        for class_ in cls_.mro():
            if hasattr(class_, "_TASK_CLASSES") and class_ is not cls_:
                tasks_in_hierarchy.extend(class_._TASK_CLASSES)

        if tasks_in_hierarchy:
            task_classes += tasks_in_hierarchy

        # Make sure the tasks are unique. Note that the order here is important
        # so that iterating the list of tasks is deterministic, unique, and the
        # tasks given in the class' module list are shown before tasks inherited
        # from parent classes.
        cls_._TASK_CLASSES = []
        for task in task_classes:
            if task not in cls_._TASK_CLASSES:
                cls_._TASK_CLASSES.append(task)

        # If no backend support described in the class, add current backend
        # as the only backend that can load models trained by this module
        cls_.SUPPORTED_LOAD_BACKENDS = getattr(
            cls_, SUPPORTED_LOAD_BACKENDS_VAR_NAME, [backend_type]
        )

        # Set its own backend_type as an attribute
        cls_.BACKEND_TYPE = backend_type

        # Verify UUID and add this module to the module registry
        if not backend_module_impl:
            if cls_.MODULE_ID in module_registry():
                error(
                    "<COR30607646E>",
                    RuntimeError(
                        "MODULE_ID `{}` conflicts for classes `{}` and `{}`".format(
                            cls_.MODULE_ID,
                            cls_.__name__,
                            module_registry()[cls_.MODULE_ID].__name__,
                        )
                    ),
                )
            module_registry()[cls_.MODULE_ID] = cls_

        # Register backend
        _register_module_implementation(
            cls_, backend_type, cls_.MODULE_ID, backend_config_override
        )

        return cls_

    return decorator


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
        implementation_class (type): The class that is used to implement this
            backend type for the given module_id
        backend_type (str): Value from MODULE_BACKEND_TYPES that indicates the
            backend that this class implements
        module_id (str): The module_id from the caikit.core module registry that
            this class overloads
        backend_config_override (Dict): Dictionary containing essential
            overrides for the backend config. This will get stored with the
            implementation_class class name and will automatically get picked up
            and merged with other such configs for a specific backend

    """

    backend_config_override = backend_config_override or {}

    log.debug(
        "Registering backend [%s] implementation for module [%s]",
        backend_type,
        module_id,
    )

    error.value_check(
        "<COR86780140E>",
        backend_type in module_backend_types(),
        "Cannot override implementation of {} for unknown backend type {}",
        module_id,
        backend_type,
    )

    core_class = module_registry().get(module_id)
    if core_class is None:
        # TODO! Inject a dummy entry that will raise on usage
        # Info level log: this is normal behavior if backend module is imported
        # before the base module
        log.info(
            "<COR86369940I>",
            "No core class found for new backend %s with module ID %s",
            implementation_class.__name__,
            module_id,
        )

    # Do the registration!
    module_type_mapping = module_backend_registry().setdefault(module_id, {})

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
