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
import alog

# Local
from .rpcs import CaikitRPCBase, ModuleClassTrainRPC, TaskPredictRPC
from caikit.core import ModuleBase, TaskBase
from caikit.core.signature_parsing.module_signature import CaikitMethodSignature

log = alog.use_channel("CREATE-RPCS")

## Globals #####################################################################

TRAIN_FUNCTION_NAME = "train"

## Utilities ###################################################################


def create_inference_rpcs(modules: List[Type[ModuleBase]]) -> List[CaikitRPCBase]:
    """Handles the logic to create all the RPCs for inference"""
    rpcs = []
    task_groups = _group_modules_by_task(modules)

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

    return rpcs


def create_training_rpcs(modules: List[Type[ModuleBase]]) -> List[CaikitRPCBase]:
    """Handles the logic to create all the RPCs for training"""

    rpcs = []

    for ck_module in modules:
        if not ck_module.TASK_CLASS:
            log.debug("Skipping module %s with no task", ck_module)
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
    return rpcs


def _group_modules_by_task(
    modules: List[Type[ModuleBase]],
) -> Dict[Type[TaskBase], List[CaikitMethodSignature]]:
    task_groups = {}
    for ck_module in modules:
        if ck_module.TASK_CLASS:
            ck_module_task_name = ck_module.TASK_CLASS.__name__
            if ck_module_task_name is not None:
                for (
                    input_streaming,
                    output_streaming,
                    signature,
                ) in ck_module._INFERENCE_SIGNATURES:
                    task_groups.setdefault(ck_module.TASK_CLASS, {}).setdefault(
                        (input_streaming, output_streaming), []
                    ).append(signature)
    return task_groups
