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
This script auto-generates the `caikit-runtime.proto` RPC definitions for a
collection of caikit.core derived libraries
"""

# Standard
from enum import Enum
from typing import List, Type

# First Party
import alog

# Local
from ... import get_config
from .core_module_helpers import get_module_info
from .primitives import is_primitive_method
from .rpcs import CaikitRPCBase, ModuleClassTrainRPC, TaskPredictRPC
from .signature_parsing.module_signature import CaikitCoreModuleMethodSignature
from caikit.core import ModuleBase

log = alog.use_channel("CREATE-RPCS")

## Globals #####################################################################

INFERENCE_FUNCTION_NAME = "run"
TRAIN_FUNCTION_NAME = "train"

## Utilities ###################################################################


class ServiceType(Enum):
    INFERENCE = 1
    TRAINING = 2


def create_inference_rpcs(modules: List[Type[ModuleBase]]) -> List[CaikitRPCBase]:
    """Handles the logic to create all the RPCs for inference"""

    primitive_data_model_types = (
        get_config().runtime.service_generation.primitive_data_model_types
    )

    rpcs = []
    # Inference specific logic:
    # Remove non-primitive modules (including modules that return None type)
    primitive_modules = _remove_non_primitive_modules(
        modules, primitive_data_model_types
    )

    # Create the RPCs for each module
    rpcs.extend(
        _create_rpcs_for_modules(
            primitive_modules, primitive_data_model_types, INFERENCE_FUNCTION_NAME
        )
    )

    return rpcs


def create_training_rpcs(modules: List[Type[ModuleBase]]) -> List[CaikitRPCBase]:
    """Handles the logic to create all the RPCs for training"""

    rpcs = []

    primitive_data_model_types = (
        get_config().runtime.service_generation.primitive_data_model_types
    )

    for ck_module in modules:
        # If this train function has not been changed from the base, skip it as
        # a module that can't be trained
        #
        # HACK alert! I'm struggling to find the right way to identify this
        #   condition, so for now, we'll use the string repr
        train_fn = getattr(ck_module, TRAIN_FUNCTION_NAME)
        if str(train_fn).startswith(f"<bound method ModuleBase.{TRAIN_FUNCTION_NAME}"):
            log.debug(
                "Skipping train API for %s with no %s function",
                ck_module,
                TRAIN_FUNCTION_NAME,
            )
            continue

        signature = CaikitCoreModuleMethodSignature(ck_module, TRAIN_FUNCTION_NAME)
        log.debug(
            "Function signature for %s::%s [%s -> %s]",
            ck_module,
            TRAIN_FUNCTION_NAME,
            signature.parameters,
            signature.return_type,
        )
        with alog.ContextLog(log.debug, "Generating train RPC for %s", ck_module):
            try:
                rpcs.append(ModuleClassTrainRPC(signature, primitive_data_model_types))
                log.debug("Successfully generated train RPC for %s", ck_module)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log.warning(
                    "Cannot generate train rpc for %s: %s",
                    ck_module,
                    err,
                    exc_info=True,
                )
    return rpcs


def _remove_non_primitive_modules(
    modules: List[Type[ModuleBase]],
    primitive_data_model_types: List[str],
) -> List[Type[ModuleBase]]:
    primitive_modules = []
    # If the module is not "primitive" we won't include it
    for ck_module in modules:
        signature = CaikitCoreModuleMethodSignature(ck_module, "run")
        if signature.parameters and signature.return_type:
            if not is_primitive_method(signature, primitive_data_model_types):
                log.debug("Skipping non-primitive module %s", ck_module)
                continue

            primitive_modules.append(ck_module)
    return primitive_modules


def _create_rpcs_for_modules(
    modules: List[Type[ModuleBase]],
    primitive_data_model_types: List[str],
    fname: str = INFERENCE_FUNCTION_NAME,
) -> List[CaikitRPCBase]:
    """Create the RPCs for each module"""
    rpcs = []
    task_groups = {}

    for ck_module in modules:
        module_info = get_module_info(ck_module)
        signature = CaikitCoreModuleMethodSignature(ck_module, fname)
        # Group each module by its task
        if module_info is not None:
            task_groups.setdefault((module_info.library, module_info.type), []).append(
                signature
            )

    # Create the RPC for each task
    for task, task_methods in task_groups.items():
        with alog.ContextLog(log.debug, "Generating task RPC for %s", task):
            try:
                rpcs.append(
                    TaskPredictRPC(task, task_methods, primitive_data_model_types)
                )
                log.debug("Successfully generated task RPC for %s", task)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log.warning(
                    "Cannot generate task rpc for %s: %s",
                    task,
                    err,
                    exc_info=True,
                )

    return rpcs
