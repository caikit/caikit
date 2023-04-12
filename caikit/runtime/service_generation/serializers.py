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
This package has classes that will serialize a python interface to a protocol buffer interface.

Typically used for `caikit.core.module`s that expose .train and .run functions.
"""
# Standard
from typing import Dict, List, Optional, Tuple, Type, get_args
import abc

# First Party
import alog

# Local
from . import primitives, type_helpers
from .compatibility_checker import ApiFieldNames
from .signature_parsing.module_signature import (
    CaikitCoreModuleMethodSignature,
    CustomSignature,
)
from caikit.core.module import ModuleBase
from caikit.interfaces.runtime.data_model import ModelPointer, TrainingJob
from caikit.runtime.service_generation.data_stream_source import make_data_stream_source

log = alog.use_channel("RPC-SERIALIZERS")

INDENT = "    "


class RPCSerializerBase(abc.ABC):
    @property
    @abc.abstractmethod
    def module_list(self) -> List[Type[ModuleBase]]:
        """Return the list of caikit.core Modules that can be invoked using this RPC"""

    @property
    @abc.abstractmethod
    def request(self) -> "_RequestMessage":
        """Return the internal representation of the request message type for this RPC"""


class ModuleClassTrainRPC(RPCSerializerBase):
    """Helper class to create a unique RPC corresponding with the train function
    for a given module class
    """

    def __init__(self, method_signature: CaikitCoreModuleMethodSignature):
        """Initialize a .proto generator with a single module to convert

        Args:
            method_signature (CaikitCoreModuleMethodSignature): The module method signature to
            generate an RPC for
        """
        self.clz: Type[ModuleBase] = method_signature.module
        self._method = ModuleClassTrainRPC._mutate_method_signature_for_training(
            method_signature
        )
        self.name = self._module_class_to_rpc_name()

        # Compute the mapping from argument name to type for the module's run
        log.debug3("Param Dict: %s", self._method.parameters)
        log.debug3("Return Type: %s", self._method.return_type)

        # Store the input and output protobuf message types for this RPC
        self.return_type = self._method.return_type
        self._req = _RequestMessage(
            self._module_class_to_req_name(), self._method.parameters
        )

    @property
    def module_list(self) -> List[Type[ModuleBase]]:
        """Returns a list containing the single caikit.core.module type that this RPC is for"""
        return [self.clz]

    @property
    def request(self) -> "_RequestMessage":
        return self._req

    def _module_class_to_rpc_name(self) -> str:
        """Helper function to convert from the name of a module to the name of the
        request RPC function
        """
        module_split = self.clz.__module__.split(".")
        return snake_to_upper_camel(
            f"{module_split[1]}_{module_split[2]}_{self.clz.__name__}_Train"
        )

    def _module_class_to_req_name(self) -> str:
        """Helper function to convert from the name of a module to the name of the
        request RPC message

        Example: self.clz._module__ = sample_lib.blocks.sample_task.sample_implementation

        return: BlocksSampleTaskSampleBlockTrainRequest

        """
        module_split = self.clz.__module__.split(".")
        return snake_to_upper_camel(
            f"{module_split[1]}_{module_split[2]}_{self.clz.__name__}_TrainRequest"
        )

    @staticmethod
    def _mutate_method_signature_for_training(
        signature,
    ) -> Optional[CaikitCoreModuleMethodSignature]:
        # Change return type for async training interface
        return_type = TrainingJob

        new_params = {"model_name": str}
        for name, typ in signature.parameters.items():
            if type_helpers.has_data_stream(typ):
                # Assume this is training data
                # Save just the stream for now
                stream_type = type_helpers.get_data_stream_type(typ)
                element_types = get_args(stream_type)
                assert (
                    len(element_types) == 1
                ), "Cannot handle DataStream with multiple type args"
                element_type = element_types[0]
                new_params[name] = make_data_stream_source(element_type)
            elif type_helpers.is_model_type(typ):
                # Found a model pointer
                new_params[name] = ModelPointer
            else:
                new_params[name] = typ

        return CustomSignature(
            original_signature=signature, parameters=new_params, return_type=return_type
        )


class TaskPredictRPC(RPCSerializerBase):
    """Helper class to create a unique RPC for the aggregate set of Modules that
    implement the same task
    """

    def __init__(
        self,
        task: Tuple[str, str],
        method_signatures: List[CaikitCoreModuleMethodSignature],
        primitive_data_model_types: List[str],
    ):
        """Initialize a .proto generator with all modules of a given task to convert

        Args:
            task (Tuple[str, str]): The library / ai-problem-task combo that describes the task
                type. For example: ("my_caikit_library", "classification")

            method_signatures (List[CaikitCoreModuleMethodSignature]): The list of method
                signatures from concrete modules implementing this task

            primitive_data_model_types: List[str]
                List of primitive data model types for a caikit_* library, such as
                caikit.interfaces.nlp.data_model.RawDocument for nlp domains
        """
        self.task = task
        self._module_list = [method.module for method in method_signatures]
        self._method_signatures = method_signatures

        # Aggregate the argument signature types into a single parameters_dict
        parameters_dict = {}
        for method in method_signatures:
            primitive_arg_dict = primitives.to_primitive_signature(
                method.parameters, primitive_data_model_types
            )
            for arg_name, arg_type in primitive_arg_dict.items():
                current_val = parameters_dict.get(arg_name, arg_type)
                # TODO: raise runtime error here!
                # TODO: need to resolve Optional[T] vs. T - if both come in, use Optional[T]
                assert (
                    current_val == arg_type
                ), f"Conflicting value types for arg {arg_name}: {current_val} != {arg_type}"

                parameters_dict[arg_name] = arg_type

        self._req = _RequestMessage(self._task_to_req_name(), parameters_dict)

        # Validate that the return_type of all modules in the grouping matches
        return_types = {method.return_type for method in method_signatures}
        assert len(return_types) == 1, f"Found multiple return types for task [{task}]"
        return_type = list(return_types)[0]
        self.return_type = return_type

        # Create the rpc name based on the module type
        self.name = self._task_to_rpc_name()

    @property
    def module_list(self) -> List[Type[ModuleBase]]:
        """Returns the list of all caikit.core.modules that this RPC will be for. These should all
        be of the same ai-problem, e.g. my_caikit_library.[blocks | workflows].classification"""
        return self._module_list

    @property
    def request(self) -> "_RequestMessage":
        return self._req

    def _task_to_req_name(self) -> str:
        """Helper function to convert the pair of library name and task name to
        a request message name
        """
        return snake_to_upper_camel(f"{self.task[1]}_Request")

    def _task_to_rpc_name(self) -> str:
        """Helper function to convert the pair of library name and task name
        to an RPC name

        Example: self.task = (sample_lib, sample_task)

        return: SampleTaskPredict
        """

        return snake_to_upper_camel(f"{self.task[1]}_Predict")


class _RequestMessage:
    """Helper class to create the input request message that wraps up the inputs
    for a given function. The request Contains N named data-model or primitive
    objects.
    """

    def __init__(self, msg_name: str, params: Dict[str, Type]):
        """Initialize with the module class and the parsed parameters to the run
        function.
        """
        self.name = msg_name
        self.triples = []

        existing_fields = ApiFieldNames.get_fields_for_message(self.name)

        if len(existing_fields) > 0:
            last_used_number = max(existing_fields.values())
        else:
            last_used_number = 0

        for _, (item_name, p) in enumerate(params.items()):
            if item_name in existing_fields:
                # if field existed previously, get the original number from there
                num = existing_fields[item_name]
            else:
                num = last_used_number + 1
                last_used_number += 1
            self.triples.append((p, item_name, num))

        self.triples.sort(key=lambda x: x[2])


def snake_to_upper_camel(string: str) -> str:
    """Simple snake -> upper camel conversion"""
    return "".join([part[0].upper() + part[1:] for part in string.split("_")])
