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
from typing import Dict, List, Type

# First Party
from aconfig import aconfig
import alog

# Local
from .rpcs import CaikitRPCBase, ModuleClassTrainRPC, TaskPredictRPC
from caikit.core import ModuleBase, TaskBase
from caikit.core.exceptions import error_handler
from caikit.core.signature_parsing.module_signature import CaikitMethodSignature

log = alog.use_channel("CREATE-RPCS")
error = error_handler.get(log)

## Globals #####################################################################

TRAIN_FUNCTION_NAME = "train"

## Utilities ###################################################################


def assert_compatible(modules: List[str], previous_modules: List[str]):
    """Logic about whether it's okay to include this set of modules in service generation

    Args:
        modules: list of module IDs that we are considering in service generation
        previous_modules: list of module IDs that were supported in the previous service version

    Raises:
        If a new service should not be built with this set of modules
    """
    regressed_modules = set(previous_modules) - set(modules)
    if len(regressed_modules) > 0:
        log.error(
            "BREAKING CHANGE FOUND! These modules became unsupported. These models were "
            "on the supported list in previous version, but now are no longer supported."
        )
        for mod in regressed_modules:
            log.error("Regressed module: %s", mod)

    error.value_check(
        "<SVC68235724E>",
        len(regressed_modules) == 0,
        "BREAKING CHANGE! Found unsupported module(s) that were previously supported: {}",
        regressed_modules,
    )


def create_inference_rpcs(
    modules: List[Type[ModuleBase]], caikit_config: aconfig.Config = None
) -> List[CaikitRPCBase]:
    """Handles the logic to create all the RPCs for inference"""
    rpcs = []

    included_task_types = (
        caikit_config
        and caikit_config.runtime.service_generation
        and caikit_config.runtime.service_generation.task_types
        and caikit_config.runtime.service_generation.task_types.included
    ) or []

    excluded_task_types = (
        caikit_config
        and caikit_config.runtime.service_generation
        and caikit_config.runtime.service_generation.task_types
        and caikit_config.runtime.service_generation.task_types.excluded
    ) or []

    task_groups = _group_modules_by_task(
        modules, included_task_types, excluded_task_types
    )

    # Create the RPC for each task
    for task, task_methods in task_groups.items():
        with alog.ContextLog(log.debug, "Generating task RPC for %s", task):
            for streaming_type, method_signatures in task_methods.items():
                input_streaming, output_streaming = streaming_type
                try:
                    rpcs.append(
                        TaskPredictRPC(
                            task, method_signatures, input_streaming, output_streaming
                        )
                    )
                    log.debug("Successfully generated task RPC for %s", task)
                except Exception as err:  # pylint: disable=broad-exception-caught
                    log.warning(
                        "Cannot generate task rpc for %s: %s",
                        task,
                        err,
                        exc_info=True,
                    )

    return sorted(rpcs, key=lambda x: x.name)


def create_training_rpcs(modules: List[Type[ModuleBase]]) -> List[CaikitRPCBase]:
    """Handles the logic to create all the RPCs for training"""

    rpcs = []

    for ck_module in modules:
        if not ck_module.tasks:
            log.debug("Skipping module %s with no tasks", ck_module)
            continue

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

        signature = ck_module.TRAIN_SIGNATURE
        log.debug(
            "Function signature for %s::%s [%s -> %s]",
            ck_module,
            TRAIN_FUNCTION_NAME,
            signature.parameters,
            signature.return_type,
        )
        with alog.ContextLog(log.debug, "Generating train RPC for %s", ck_module):
            try:
                rpcs.append(ModuleClassTrainRPC(signature))
                log.debug("Successfully generated train RPC for %s", ck_module)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log.warning(
                    "Cannot generate train rpc for %s: %s",
                    ck_module,
                    err,
                    exc_info=True,
                )
    return sorted(rpcs, key=lambda x: x.name)


def _group_modules_by_task(
    modules: List[Type[ModuleBase]],
    included_tasks: List[Type[TaskBase]],
    excluded_tasks: List[Type[TaskBase]],
) -> Dict[Type[TaskBase], List[CaikitMethodSignature]]:
    task_groups = {}
    # Sort modules so the order of modules processed is deterministic
    modules = sorted(modules, key=lambda x: x.MODULE_ID)
    for ck_module in modules:
        for task_class in ck_module.tasks:
            if (
                included_tasks
                and task_class.__name__ not in included_tasks
                or excluded_tasks
                and task_class.__name__ in excluded_tasks
            ):
                continue

            ck_module_task_name = task_class.__name__
            if ck_module_task_name is not None:
                for (
                    input_streaming,
                    output_streaming,
                    signature,
                ) in ck_module.get_inference_signatures(task_class):
                    task_groups.setdefault(task_class, {}).setdefault(
                        (input_streaming, output_streaming), []
                    ).append(signature)
    return task_groups
