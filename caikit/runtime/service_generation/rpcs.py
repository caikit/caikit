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
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin
import abc
import copy
import typing

# First Party
from py_to_proto.dataclass_to_proto import (  # NOTE: Imported from here for compatibility
    Annotated,
    FieldNumber,
)
import alog

# Local
from ...interfaces.common.data_model.stream_sources import S3Path
from . import protoable, type_helpers
from .compatibility_checker import ApiFieldNames
from .data_stream_source import make_data_stream_source
from caikit.core import ModuleBase, TaskBase
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import make_dataobject
from caikit.core.signature_parsing import CaikitMethodSignature, CustomSignature
from caikit.interfaces.runtime.data_model import ModelPointer, TrainingJob
from caikit.runtime.names import (
    get_task_predict_request_name,
    get_task_predict_rpc_name,
    get_train_parameter_name,
    get_train_request_name,
    get_train_rpc_name,
)

log = alog.use_channel("RPC-SERIALIZERS")

INDENT = "    "


class CaikitRPCBase(abc.ABC):
    @property
    @abc.abstractmethod
    def module_list(self) -> List[Type[ModuleBase]]:
        """Return the list of caikit.core Modules that can be invoked using this RPC"""

    @property
    @abc.abstractmethod
    def request(self) -> "_RequestMessage":
        """Return the internal representation of the request message type for this RPC"""

    @property
    def name(self) -> str:
        return self._name

    def create_request_data_model(self, package_name: str) -> Type[DataBase]:
        """Dynamically create data model for this RPC's request message"""
        return self.request.create_data_model(package_name)

    def create_rpc_json(self, package_name: str) -> Dict:
        """Return json snippet for the service definition of this RPC"""
        rpc_json = {
            "name": f"{self.name}",
            "input_type": f"{package_name}.{self.request.name}",
            "output_type": self.return_type.get_proto_class().DESCRIPTOR.full_name,
        }
        return rpc_json


class ModuleClassTrainRPC(CaikitRPCBase):
    """Helper class to create a unique RPC corresponding with the train function
    for a given module class
    """

    def __init__(
        self,
        method_signature: CaikitMethodSignature,
    ):
        """Initialize a .proto generator with a single module to convert

        Args:
            method_signature (CaikitMethodSignature): The module method signature to
            generate an RPC for
        """
        self.clz: Type[ModuleBase] = method_signature.module
        self._method = ModuleClassTrainRPC._mutate_method_signature_for_training(
            method_signature
        )
        self._name = ModuleClassTrainRPC.module_class_to_rpc_name(self.clz)

        # Compute the mapping from argument name to type for the module's run
        log.debug3("Param Dict: %s", self._method.parameters)
        log.debug3("Return Type: %s", self._method.return_type)

        # Store the input and output protobuf message types for this RPC
        self.return_type = self._method.return_type

        # Inner / outer requests
        # The actual training parameters from the method signature are nested under `parameters`
        self._inner_request = _RequestMessage(
            ModuleClassTrainRPC.module_class_to_inner_request_name(self.clz),
            self._method.parameters,
            self._method.default_parameters,
        )

        params = {"model_name": str, "output_path": S3Path, "parameters": "PLACEHOLDER"}

        self._req = _RequestMessage(
            ModuleClassTrainRPC.module_class_to_req_name(self.clz),
            params,
            {},
        )

    def create_request_data_model(self, package_name: str):
        """Partial override of CaikitRPCBase.create_request_data_model to take care of the inner
        request data model"""
        # Build the inner request data model
        inner_request_data_model = self._inner_request.create_data_model(package_name)
        # Insert the new type into the outer request
        for triple_index, _ in enumerate(self._req.triples):
            if self._req.triples[triple_index][1] == "parameters":
                triple = self._req.triples[triple_index]
                self._req.triples[triple_index] = (
                    inner_request_data_model,
                    triple[1],
                    triple[2],
                )
        # Call the base class' method
        return CaikitRPCBase.create_request_data_model(
            self=self, package_name=package_name
        )

    @property
    def module_list(self) -> List[Type[ModuleBase]]:
        """Returns a list containing the single caikit.core.module type that this RPC is for"""
        return [self.clz]

    @property
    def request(self) -> "_RequestMessage":
        return self._req

    @staticmethod
    def module_class_to_rpc_name(module_class: Type[ModuleBase]) -> str:
        """Helper function to convert from the name of a module to the name of the
        request RPC function
        """
        return get_train_rpc_name(module_class)

    @staticmethod
    def module_class_to_req_name(module_class: Type[ModuleBase]) -> str:
        """Helper function to convert from the name of a module to the name of the
        request RPC message

        Example: self.clz._module__ = sample_lib.modules.sample_task.sample_implementation

        return: SampleTaskSampleModuleTrainRequest

        """
        return get_train_request_name(module_class)

    @staticmethod
    def module_class_to_inner_request_name(module_class: Type[ModuleBase]) -> str:
        """Helper function to convert from a module to the name of the inner message
        containing all the training parameters

        Example: self.clz._module__ = sample_lib.modules.sample_task.sample_implementation

        return: SampleTaskSampleModuleTrainParameters
        """
        return get_train_parameter_name(module_class)

    @staticmethod
    def _mutate_method_signature_for_training(
        signature: CaikitMethodSignature,
    ) -> Optional[CaikitMethodSignature]:
        # Change return type for async training interface
        return_type = TrainingJob

        # Loop through params and edit as needed
        new_params = {}
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
                new_params[name] = protoable.handle_protoables_in_union(
                    field_name=name,
                    arg_type=typ,
                )

        return CustomSignature(
            original_signature=signature, parameters=new_params, return_type=return_type
        )


class TaskPredictRPC(CaikitRPCBase):
    """Helper class to create a unique RPC for the aggregate set of Modules that
    implement the same task
    """

    def __init__(
        self,
        task: Type[TaskBase],
        method_signatures: List[CaikitMethodSignature],
        input_streaming: bool = False,
        output_streaming: bool = False,
    ):
        """Initialize a .proto generator with all modules of a given task to convert

        Args:
            task (Type[TaskBase]): Task type

            method_signatures (List[CaikitMethodSignature]): The list of method
                signatures from concrete modules implementing this task
        """
        self.task = task
        self._method_signatures = method_signatures
        self._input_streaming = input_streaming
        self._output_streaming = output_streaming

        # Aggregate the argument signature types into a single parameters_dict
        parameters_dict = {}
        default_parameters = {}
        for method in method_signatures:
            default_parameters.update(method.default_parameters)
            new_params = self._handle_task_inputs(method.parameters)
            primitive_arg_dict = protoable.to_protoable_signature(new_params)
            for arg_name, arg_type in primitive_arg_dict.items():
                current_val = parameters_dict.get(arg_name, arg_type)
                # TODO: raise runtime error here instead of assert!
                # TODO: need to resolve Optional[T] vs. T - if both come in, use Optional[T]
                assert (
                    current_val == arg_type
                ), f"Conflicting value types for arg {arg_name}: {current_val} != {arg_type}"

                parameters_dict[arg_name] = arg_type

        self._req = _RequestMessage(
            self._task_to_req_name(), parameters_dict, default_parameters
        )

        # Validate that the return_type of all modules in the grouping matches
        return_types = {
            protoable.get_protoable_return_type(method.return_type)
            for method in method_signatures
        }
        assert len(return_types) == 1, (
            f"Found multiple return types for task [{task}], rpc: [{self._task_to_rpc_name()}. "
            f"Return types: {return_types}]"
        )
        self.return_type = list(return_types)[0]

        # Create the rpc name based on the module type
        self._name = self._task_to_rpc_name()

    @property
    def module_list(self) -> List[Type[ModuleBase]]:
        """Returns the list of all caikit.core.modules that this RPC will be for. These should all
        be of the same ai-problem, e.g. my_caikit_library.modules.classification
        """
        return [method.module for method in self._method_signatures]

    @property
    def request(self) -> "_RequestMessage":
        return self._req

    @property
    def input_streaming(self) -> bool:
        return self._input_streaming

    @property
    def output_streaming(self) -> bool:
        return self._output_streaming

    def create_rpc_json(self, package_name: str) -> Dict:
        """Return json snippet for the service definition of this RPC"""
        if self.output_streaming:
            output_type_name = (
                typing.get_args(self.return_type)[0]
                .get_proto_class()
                .DESCRIPTOR.full_name
            )
        else:
            output_type_name = self.return_type.get_proto_class().DESCRIPTOR.full_name

        rpc_json = {
            "name": f"{self.name}",
            "input_type": f"{package_name}.{self.request.name}",
            "output_type": output_type_name,
            "server_streaming": self.output_streaming,
            "client_streaming": self.input_streaming,
        }
        return rpc_json

    def _handle_task_inputs(self, method_params: Dict[str, Any]) -> Dict[str, Any]:
        """Overrides input params with types specified in the Task"""
        new_params = {}
        if self._input_streaming:
            req_params = self.task.get_required_parameters(input_streaming=True)
            for param_name, param_type in method_params.items():
                if param_name in req_params:
                    # double check this condition, although it should already have been validated
                    # also assuming both are iterables
                    if get_args(req_params[param_name])[0] in get_args(param_type):
                        new_params[param_name] = get_args(req_params[param_name])[0]
                else:
                    new_params[param_name] = param_type
            return new_params
        # for unary input cases
        req_params = self.task.get_required_parameters(input_streaming=False)
        for param_name, param_type in method_params.items():
            new_params[param_name] = req_params.get(param_name, param_type)
        return new_params

    def _task_to_req_name(self) -> str:
        """Helper function to convert the pair of library name and task name to
        a request message name
        """
        return get_task_predict_request_name(
            self.task, self._input_streaming, self._output_streaming
        )

    def _task_to_rpc_name(self) -> str:
        """Helper function to convert the pair of library name and task name
        to an RPC name

        Example: self.task = (sample_lib, sample_task)

        return: SampleTaskPredict
        """
        return get_task_predict_rpc_name(
            self.task, self._input_streaming, self._output_streaming
        )


class _RequestMessage:
    """Helper class to create the input request message that wraps up the inputs
    for a given function. The request Contains N named data-model or primitive
    objects.
    """

    def __init__(
        self, msg_name: str, params: Dict[str, Type], default_map: Dict[str, Any]
    ):
        """Initialize with the module class and the parsed parameters to the run
        function.
        """
        self.name = msg_name
        self.triples = []
        self.default_map = default_map

        existing_fields = ApiFieldNames.get_fields_for_message(self.name)

        last_used_number = max(existing_fields.values()) if existing_fields else 0

        for _, (item_name, typ) in enumerate(params.items()):
            if item_name in existing_fields:
                # if field existed previously, get the original number from there
                num = existing_fields[item_name]
            else:
                num = last_used_number + 1
                if get_origin(typ) is Union:
                    last_used_number += len(get_args(typ))
                else:
                    last_used_number += 1
            self.triples.append((typ, item_name, num))

        self.triples.sort(key=lambda x: x[2])

    def create_data_model(self, package_name: str) -> Type[DataBase]:
        """Dynamically create a data model for this request message"""
        properties = {
            # triple e.g. ('caikit.interfaces.common.ProducerPriority', 'producer_id', 1)
            # This does not take care of nested descriptors
            triple[1]: Annotated[triple[0], FieldNumber(triple[2])]
            for triple in self.triples
            if triple[1] not in self.default_map
        }
        optional_properties = {
            triple[1]: Annotated[Optional[triple[0]], FieldNumber(triple[2])]
            for triple in self.triples
            if triple[1] in self.default_map
        }
        attrs = copy.copy(self.default_map)

        if not properties and not optional_properties:
            log.warning(
                "No arguments found for request %s. Cannot generate rpc",
                self.name,
            )
            return None

        return make_dataobject(
            package=package_name,
            name=self.name,
            attrs=attrs,
            annotations={**properties, **optional_properties},
        )
