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
from typing import Any, Dict, List, Tuple, Type, Union, get_args, get_origin
import inspect

# First Party
import alog

# Local
from caikit.core.exceptions import error_handler
from caikit.core.modules.base import ModuleBase
from caikit.core.modules.config import ModuleConfig
from caikit.core.modules.meta import _ModuleBaseMeta
from caikit.core.registries import module_registry
from caikit.core.task import TaskBase
from caikit.interfaces.common.data_model.remote import ConnectionInfo
from caikit.interfaces.runtime.service import (
    CaikitRPCDescriptor,
    get_task_predict_request_name,
    get_task_predict_rpc_name,
    get_train_request_name,
    get_train_rpc_name,
)

log = alog.use_channel("REM_MODULE_CFG")
error = error_handler.get(log)


### Remote Module Config


class RemoteModuleConfig(ModuleConfig):
    """Helper class to differentiate a local ModuleConfig and a RemoteModuleConfig. The structure
    should contain the following fields/structure"""

    # Remote information for how to access the server.
    connection: ConnectionInfo
    protocol: str
    model_key: str

    # Method Information
    # use list and tuples instead of a dictionary to avoid aconfig.Config error
    task_methods: List[Tuple[Type[TaskBase], List[CaikitRPCDescriptor]]]
    train_method: CaikitRPCDescriptor

    # Target Module Information
    module_id: str
    module_name: str
    model_path: str

    # Reset reserved_keys, so we can manually add module_path
    reserved_keys = []

    @classmethod
    def load_from_module(
        cls,
        module_reference: Union[str, Type[ModuleBase], ModuleBase],
        connection_info: ConnectionInfo,
        protocol: str,
        model_key: str,
        model_path: str,
    ) -> "RemoteModuleConfig":
        """Construct a new remote module configuration from an existing local Module

        Args:
            module_reference: Union[str, Type[ModuleBase]]:
                Module_reference should be one of the following: the id of the locally loaded module,
                ,or a module class

            model_path (str):
                The path used to load this module

            connection_info ConnectionInfo:
                The connection information of the remote to use

            protocol: str
                The protocol to connect with

            model_key: str
                The model key to use when sending GRPC requests. An example is mm-model-id

           Returns:
               model_config (RemoteModuleConfig): Instantiated RemoteModuleConfig for
               model given model_path.
        """

        # Validate model path arg
        error.type_check("<COR71170339E>", str, model_path=model_path)

        # Get local module reference
        error.type_check(
            "<COR71270339E>",
            str,
            ModuleBase,
            _ModuleBaseMeta,
            module_reference=module_reference,
        )
        if isinstance(module_reference, ModuleBase) or (
            inspect.isclass(module_reference)
            and issubclass(module_reference, ModuleBase)
        ):
            local_module_class = module_reference
        else:
            if module_reference not in module_registry():
                raise KeyError(f"Unknown module reference {module_reference}")

            local_module_class = module_registry().get(module_reference)

        # Construct model config dict
        remote_config_dict = {
            # Connection info
            "connection": connection_info,
            "protocol": protocol,
            "model_key": model_key,
            # Method info
            "task_methods": [],
            "train_method": None,
            # Source module info
            "model_path": model_path,
            "module_id": f"{local_module_class.MODULE_ID}-remote",
            "module_name": f"{local_module_class.MODULE_NAME} Remote",
        }

        # Parse inference methods signatures
        for task_class in local_module_class.tasks:
            task_methods = []
            for input, output, signature in local_module_class.get_inference_signatures(
                task_class
            ):

                # Don't get the actual DataBaseObject as the ServicePackage might not have
                # been generated
                request_class_name = get_task_predict_request_name(
                    task_class, input, output
                )
                task_request_name = get_task_predict_rpc_name(task_class, input, output)

                if hasattr(signature.return_type, "__name__"):
                    task_return_type = signature.return_type.__name__
                else:
                    task_return_type = get_origin(signature.return_type).__name__

                # Get the underlying DataBaseObject for stream types
                if output and get_args(signature.return_type) != ():
                    task_return_type = get_args(signature.return_type)[0].__name__

                # Generate the rpc name and task type
                task_methods.append(
                    CaikitRPCDescriptor(
                        signature=signature,
                        request_dm_name=request_class_name,
                        response_dm_name=task_return_type,
                        rpc_name=task_request_name,
                        input_streaming=input,
                        output_streaming=output,
                    )
                )

            remote_config_dict["task_methods"].append((task_class, task_methods))

        # parse train method signature if there is one
        if local_module_class.TRAIN_SIGNATURE and (
            local_module_class.TRAIN_SIGNATURE.return_type is not None
            and local_module_class.TRAIN_SIGNATURE.parameters is not None
        ):
            train_request_name = get_train_request_name(local_module_class)
            train_rpc_name = get_train_rpc_name(local_module_class)

            remote_config_dict["train_method"] = CaikitRPCDescriptor(
                signature=local_module_class.TRAIN_SIGNATURE,
                request_dm_name=train_request_name,
                response_dm_name=local_module_class.TRAIN_SIGNATURE.return_type.__name__,
                rpc_name=train_rpc_name,
            )

        return RemoteModuleConfig(remote_config_dict)
