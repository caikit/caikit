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
from types import ModuleType

# Third Party
import pytest

# Local
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.conftest import temp_config
import caikit


# 🌶️🌶️🌶️🌶️🌶️
# Fixtures related to the protoc-compiled service interfaces created by `test/__init__.py`
# These compiled fixtures should be deleted with a move to `json-to-service`d service interfaces
@pytest.fixture
def compiled_caikit_runtime_inference_pb2() -> ModuleType:
    return ServicePackageFactory._get_compiled_proto_module("caikit_runtime_pb2")


@pytest.fixture
def compiled_caikit_runtime_inference_pb2_grpc() -> ModuleType:
    return ServicePackageFactory._get_compiled_proto_module("caikit_runtime_pb2_grpc")


@pytest.fixture
def compiled_caikit_runtime_train_pb2() -> ModuleType:
    return ServicePackageFactory._get_compiled_proto_module("caikit_runtime_train_pb2")


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


def test_service_package_raises_for_compiled_training_management():
    with pytest.raises(
        CaikitRuntimeException,
        match="Not allowed to get Training Management services from compiled packages",
    ):
        ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
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


def test_get_service_descriptor_raises_if_pb2_has_no_desc():
    with pytest.raises(
        CaikitRuntimeException,
        match="Could not find service descriptor in caikit_runtime_pb2",
    ):
        ServicePackageFactory._get_service_descriptor(None, "SampleLib")


def test_inference_service_registration_function(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2_grpc
):
    assert (
        compiled_inference_service.registration_function
        == compiled_caikit_runtime_inference_pb2_grpc.add_SampleLibServiceServicer_to_server
    )


def test_get_servicer_function_raises_if_grpc_has_no_function():
    with pytest.raises(
        CaikitRuntimeException,
        match="Could not find servicer function in caikit_runtime_pb2_grpc",
    ):
        ServicePackageFactory._get_servicer_function(None, "samplelib")


def test_inference_service_class(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2_grpc
):
    assert (
        compiled_caikit_runtime_inference_pb2_grpc.SampleLibServiceServicer
        == compiled_inference_service.service
    )


def test_get_servicer_class_raises_if_grpc_has_no_class():
    with pytest.raises(CaikitRuntimeException, match="Could not find servicer class"):
        ServicePackageFactory._get_servicer_class(None, "samplelib")


def test_inference_client_stub(
    compiled_inference_service, compiled_caikit_runtime_inference_pb2_grpc
):
    assert (
        compiled_caikit_runtime_inference_pb2_grpc.SampleLibServiceStub
        == compiled_inference_service.stub_class
    )


def test_get_servicer_stub_raises_if_grpc_has_no_stub():
    with pytest.raises(
        CaikitRuntimeException,
        match="Could not find servicer stub in caikit_runtime_pb2_grpc",
    ):
        ServicePackageFactory._get_servicer_stub(None, "SampleLib")


### _get_service_proto_module #############################################################
# Tests already existed - thus testing private method


def test_invalid_get_compiled_proto_module_dir():
    """If an invalid directory is provided, should throw module not found"""

    with temp_config(
        {"runtime": {"compiled_proto_module_dir": "invalid_proto_module_dir"}}
    ):
        with pytest.raises(ModuleNotFoundError):
            ServicePackageFactory._get_compiled_proto_module("caikit_runtime_pb2")


def test_invalid_get_compiled_proto_module():
    """If an invalid module is provided to _get_compiled_proto_module, it throws a ValueError"""

    with temp_config(
        {"runtime": {"compiled_proto_module_dir": "tests.fixtures.protobufs"}}
    ):
        with pytest.raises(CaikitRuntimeException):
            ServicePackageFactory._get_compiled_proto_module("invalid_module")


def test_get_compiled_proto_module():
    """If caikit_runtime_pb2 is provided to _get_compiled_proto_module, the module is returned"""
    # Local
    from tests.fixtures.protobufs import caikit_runtime_pb2

    with temp_config(
        {"runtime": {"compiled_proto_module_dir": "tests.fixtures.protobufs"}}
    ):
        service_proto_module = ServicePackageFactory._get_compiled_proto_module(
            "caikit_runtime_pb2"
        )
        assert service_proto_module == caikit_runtime_pb2


# Local
import sample_lib

MODULE_LIST = [
    module_class
    for module_class in caikit.core.registries.module_registry().values()
    if module_class.__module__.partition(".")[0] == "sample_lib"
]

### Test ServicePackageFactory._get_and_filter_modules function
def test_get_and_filter_modules_respects_excluded_task_type():
    with temp_config(
        {
            "runtime": {
                "service_generation": {"task_types": {"excluded": ["sample_task"]}}
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "sample_task" not in str(clean_modules)


def test_get_and_filter_modules_respects_excluded_modules():
    assert "InnerModule" in str(MODULE_LIST)
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {
                        "excluded": ["00110203-baad-beef-0809-0a0b02dd0e0f"]
                    }
                }
            }
        }  # excluding InnerModule
    ) as cfg:

        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "InnerModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_excluded_modules_and_excluded_task_type():
    assert "InnerModule" in str(MODULE_LIST)
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {
                        "excluded": ["00110203-baad-beef-0809-0a0b02dd0e0f"]
                    },
                    "task_types": {"excluded": ["other_task"]},
                }
            }
        }  # excluding InnerModule and OtherModule
    ) as cfg:

        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "InnerModule" not in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)
        assert "other_task" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_modules_and_included_task_types():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {
                        "included": ["00110203-baad-beef-0809-0a0b02dd0e0f"]
                    },
                    "task_types": {"included": ["other_task"]},
                }
            }
        }  # only want InnerModule and OtherModule
    ) as cfg:

        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert len(clean_modules) == 2
        assert "InnerModule" in str(clean_modules)
        assert "OtherModule" in str(clean_modules)
        assert "ListModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_modules():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {
                        "included": [
                            "00110203-baad-beef-0809-0a0b02dd0e0f",
                            "00af2203-0405-0607-0263-0a0b02dd0c2f",
                        ]
                    },
                }
            }
        }  # only want InnerModule and ListModule
    ) as cfg:

        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert len(clean_modules) == 2
        assert "InnerModule" in str(clean_modules)
        assert "ListModule" in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_task_types():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "task_types": {"included": ["sample_task"]},
                }
            }
        }  # only want sample_task which has 6 modules
    ) as cfg:

        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "InnerModule" in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_task_types_and_excluded_modules():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "task_types": {"included": ["sample_task"]},
                    "module_guids": {
                        "excluded": ["00af2203-0405-0607-0263-0a0b02dd0c2f"]
                    },
                }
            }
        }  # only want sample_task but not ListModule
    ) as cfg:

        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "InnerModule" in str(clean_modules)
        assert "ListModule" not in str(clean_modules)
