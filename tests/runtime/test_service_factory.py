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

"""Unit tests for the service factory"""
# Standard
from types import ModuleType, SimpleNamespace

# Third Party
import pytest

# Local
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit


# 🌶️🌶️🌶️🌶️🌶️
# Fixtures related to the protoc-compiled service interfaces created by `test/__init__.py`
# These compiled fixtures should be deleted with a move to `json-to-service`d service interfaces
@pytest.fixture
def compiled_caikit_runtime_inference_pb2() -> ModuleType:
    return ServicePackageFactory._get_service_proto_module("caikit_runtime_pb2")


@pytest.fixture
def compiled_caikit_runtime_inference_pb2_grpc() -> ModuleType:
    return ServicePackageFactory._get_service_proto_module("caikit_runtime_pb2_grpc")


@pytest.fixture
def compiled_caikit_runtime_train_pb2() -> ModuleType:
    return ServicePackageFactory._get_service_proto_module("caikit_runtime_train_pb2")


# /🌶️🌶️🌶️🌶️🌶️


@pytest.fixture
def compiled_inference_service() -> ServicePackage:
    return ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
        ServicePackageFactory.ServiceSource.COMPILED,
    )


@pytest.fixture
def compiled_training_service() -> ServicePackage:
    return ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
        ServicePackageFactory.ServiceSource.COMPILED,
    )


# These tests assume we have a compiled pb2 package to pull services from
def test_inference_descriptor(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2
):
    assert (
        compiled_caikit_runtime_inference_pb2._SAMPLELIBSERVICE
        == compiled_inference_service.descriptor
    )


def test_train_descriptor(compiled_training_service, compiled_caikit_runtime_train_pb2):
    assert (
        compiled_caikit_runtime_train_pb2._SAMPLELIBTRAININGSERVICE
        == compiled_training_service.descriptor
    )


# def test_get_service_descriptor_raises_if_pb2_has_no_desc(self):
#     with pytest.raises(CaikitRuntimeException):
#
#     with self.assertRaises(CaikitRuntimeException):
#         get_service_descriptor(None)


def test_inference_service_registration_function(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2_grpc
):
    assert (
        compiled_inference_service.registration_function
        == compiled_caikit_runtime_inference_pb2_grpc.add_SampleLibServiceServicer_to_server
    )


# def test_get_servicer_function_raises_if_grpc_has_no_function(self):
#     with self.assertRaises(CaikitRuntimeException):
#         get_servicer_function(None)


def test_inference_service_class(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2_grpc
):
    assert (
        compiled_caikit_runtime_inference_pb2_grpc.SampleLibServiceServicer
        == compiled_inference_service.service
    )


# def test_get_servicer_class_raises_if_grpc_has_no_class(self):
#     with self.assertRaises(CaikitRuntimeException):
#         get_servicer_class(None)


def test_inference_client_stub(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2_grpc
):
    assert (
        compiled_caikit_runtime_inference_pb2_grpc.SampleLibServiceStub
        == compiled_inference_service.stub_class
    )


# def test_get_servicer_stub_raises_if_grpc_has_no_stub(self):
#     with self.assertRaises(CaikitRuntimeException):
#         get_servicer_stub(None)


### _get_service_proto_module #############################################################
# Tests already existed - thus testing private method


def test_invalid_get_service_proto_module_dir():
    """If an invalid directory is provided, should throw module not found"""

    mock_config = SimpleNamespace(
        **{"service_proto_gen_module_dir": "invalid_proto_module_dir"}
    )
    with pytest.raises(ModuleNotFoundError):
        ServicePackageFactory._get_service_proto_module(
            "caikit_runtime_pb2", mock_config
        )


def test_invalid_get_service_proto_module():
    """If an invalid module is provided to _get_service_proto_module, it throws a ValueError"""

    mock_config = SimpleNamespace(
        **{"service_proto_gen_module_dir": "tests.fixtures.protobufs"}
    )
    with pytest.raises(CaikitRuntimeException):
        ServicePackageFactory._get_service_proto_module("invalid_module", mock_config)


def test_get_service_proto_module():
    """If caikit_runtime_pb2 is provided to _get_service_proto_module, the module is returned"""
    # Local
    from tests.fixtures.protobufs import caikit_runtime_pb2

    mock_config = SimpleNamespace(
        **{"service_proto_gen_module_dir": "tests.fixtures.protobufs"}
    )
    service_proto_module = ServicePackageFactory._get_service_proto_module(
        "caikit_runtime_pb2", mock_config
    )
    assert service_proto_module == caikit_runtime_pb2


# Third Party
import sample_lib

MODULE_LIST = [
    module_class
    for module_class in caikit.core.MODULE_REGISTRY.values()
    if module_class.__module__.partition(".")[0] == "sample_lib"
]

### Test ServicePackageFactory._remove_exclusions_from_module_list
def test_remove_exclusions_from_module_list_respects_excluded_task_type():
    assert len(MODULE_LIST) == 6  # there are 6 modules in Sample Lib
    clean_modules = ServicePackageFactory._remove_exclusions_from_module_list(
        MODULE_LIST, excluded_task_types=["sample_task"]
    )
    assert len(clean_modules) == 1
    assert "sample_task" not in str(clean_modules)


def test_remove_exclusions_from_module_list_respects_excluded_modules():
    assert "InnerBlock" in str(MODULE_LIST)
    clean_modules = ServicePackageFactory._remove_exclusions_from_module_list(
        MODULE_LIST, excluded_modules=["00110203-baad-beef-0809-0a0b02dd0e0f"]
    )  # excluding InnerBlock
    assert len(clean_modules) == 5
    assert "InnerBlock" not in str(clean_modules)


def test_remove_exclusions_from_module_list_respects_excluded_modules_and_excluded_task_type():
    assert "InnerBlock" in str(MODULE_LIST)
    assert len(MODULE_LIST) == 6  # there are 6 modules in Sample Lib
    clean_modules = ServicePackageFactory._remove_exclusions_from_module_list(
        MODULE_LIST,
        excluded_modules=["00110203-baad-beef-0809-0a0b02dd0e0f"],
        excluded_task_types=["other_task"],
    )  # excluding InnerBlock
    assert len(clean_modules) == 4
    assert "InnerBlock" not in str(clean_modules)
    assert "other_task" not in str(clean_modules)
