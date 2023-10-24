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
from pathlib import Path
import json
import os
import shutil
import tempfile
import uuid

# Third Party
from google.protobuf.message import Message
from grpc_tools import protoc
import pytest

# Local
from caikit.core.data_model import render_dataobject_protos
from caikit.runtime.dump_services import dump_grpc_services
from caikit.runtime.service_factory import (
    ServicePackage,
    ServicePackageFactory,
    get_inference_request,
    get_train_params,
    get_train_request,
)
from sample_lib import SampleModule
from sample_lib.data_model import SampleInputType, SampleOutputType
from sample_lib.data_model.sample import SampleTask
from sample_lib.modules import ListModule, OtherModule
from tests.conftest import temp_config
from tests.core.helpers import MockBackend
from tests.data_model_helpers import (
    reset_global_protobuf_registry,
    temp_dpool,
    temp_module,
)
from tests.runtime.conftest import sample_inference_service, sample_train_service
import caikit

## Helpers #####################################################################


@pytest.fixture
def clean_data_model(sample_inference_service, sample_train_service):
    """Set up a temporary descriptor pool that inherits all explicitly created
    dataobjects from the global, but skips dynamically created ones from
    datastreamsource.
    """
    with reset_global_protobuf_registry():
        with temp_dpool(
            inherit_global=True,
            skip_inherit=[".*sampletask.*\.proto"],
        ) as dpool:
            yield dpool


@pytest.fixture
def clean_data_model_with_no_sampletaskrequest_proto(
    sample_inference_service, sample_train_service
):
    """Set up a temporary descriptor pool that skips inheriting only sampletaskrequest.proto"""
    # with reset_global_protobuf_registry():
    with temp_dpool(
        inherit_global=True,
        skip_inherit=[".*sampletaskrequest.proto"],
    ) as dpool:
        yield dpool


def validate_package_with_override(
    service_type: ServicePackageFactory.ServiceType,
    domain_override: str,
    package_override: str,
) -> ServicePackage:
    """Construct and validate train or infer service package with domain and package path override

    The config should be set up for domain and/or package override, before this function is called.

    Args:
        service_type (ServicePackageFactory.ServiceType) : The type of service to build,
            only TRAINING and INFERENCE are supported
        domain_override (str): The domain to validate (maybe overridden value, or the default)
        package_override (str): The package path to validate (maybe overridden value, or the default)

    Returns: The service package that was created, for potential subsequent reuse
    """
    assert service_type in {
        ServicePackageFactory.ServiceType.TRAINING,
        ServicePackageFactory.ServiceType.INFERENCE,
    }
    svc = ServicePackageFactory.get_service_package(service_type)

    service_name = (
        f"{domain_override}TrainingService"
        if service_type == ServicePackageFactory.ServiceType.TRAINING
        else f"{domain_override}Service"
    )
    assert svc.service.__name__ == service_name
    assert svc.descriptor.full_name == f"{package_override}.{service_name}"
    for message_name in [
        x for x in svc.messages.__dict__.keys() if not x.startswith("_")
    ]:
        message: Message = getattr(svc.messages, message_name)
        assert message.DESCRIPTOR.full_name == f"{package_override}.{message_name}"

    return svc


### Private method tests #############################################################


MODULE_LIST = [
    module_class
    for module_class in caikit.core.registries.module_registry().values()
    if module_class.__module__.partition(".")[0] == "sample_lib"
]


### Test ServicePackageFactory._get_and_filter_modules function


def test_get_and_filter_modules_respects_excluded_modules():
    assert "InnerModule" in str(MODULE_LIST)
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"excluded": [ListModule.MODULE_ID]}
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            cfg, "sample_lib", False
        )
        assert "ListModule" not in str(clean_modules)
        assert "SampleModule" in str(clean_modules)
        assert "OtherModule" in str(clean_modules)


def test_get_and_filter_modules_respects_included_modules():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"included": [ListModule.MODULE_ID]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            cfg, "sample_lib", False
        )
        assert len(clean_modules) == 1
        assert "ListModule" in str(clean_modules)
        assert "SampleModule" not in str(clean_modules)


def test_assert_compatible_raises_if_prev_modules_path_is_not_valid():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "backwards_compatibility": {
                        "enabled": True,
                        "prev_modules_path": "foobar.json",  # a file that's not valid
                    }
                },
            }
        },
        "merge",
    ):
        with pytest.raises(ValueError) as e:
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.INFERENCE
            )


def test_assert_compatible_does_not_raise_if_not_enabled():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "backwards_compatibility": {
                        "enabled": False,
                    }
                },
            }
        },
        "merge",
    ):
        ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE
        )


def test_assert_compatible_raises_if_a_module_becomes_unsupported():
    with tempfile.TemporaryDirectory() as workdir:
        prev_module_file = os.path.join(workdir, "prev_modules.json")
        random_uuid = str(uuid.uuid4())
        with open(prev_module_file, "w", encoding="utf-8") as file:
            json_content = {
                "included_modules": {
                    "SampleTask": {
                        random_uuid: "<class 'sample_lib.modules.sample_task.sample_implementation.PrevSampleModule'>",
                    }
                },
            }
            file.write(json.dumps(json_content, indent=4))
        with temp_config(
            {
                "runtime": {
                    "service_generation": {
                        "backwards_compatibility": {
                            "enabled": True,
                            "prev_modules_path": prev_module_file,
                        }
                    },
                }
            },
            "merge",
        ):
            with pytest.raises(ValueError):
                # this raises because PrevSampleModule will not be included in this service generation
                ServicePackageFactory.get_service_package(
                    ServicePackageFactory.ServiceType.INFERENCE
                )


def test_assert_compatible_does_not_raise_if_supported_modules_continue_to_be_supported():
    with tempfile.TemporaryDirectory() as workdir:
        prev_module_file = os.path.join(workdir, "prev_modules.json")

        with open(prev_module_file, "w", encoding="utf-8") as file:
            json_content = {
                "included_modules": {
                    "SampleTask": {
                        "00110203-0405-0607-0809-0a0b02dd0e0f": "<class 'sample_lib.modules.sample_task.sample_implementation.SampleModule'>",
                    }
                },
            }
            file.write(json.dumps(json_content, indent=4))
        with temp_config(
            {
                "runtime": {
                    "service_generation": {
                        "backwards_compatibility": {
                            "enabled": True,
                            "prev_modules_path": prev_module_file,
                        }
                    },
                }
            },
            "merge",
        ):
            # this does not raise because SampleModule will be included in this service generation
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.INFERENCE
            )


def test_override_domain(clean_data_model):
    """
    Test override of gRPC domain generation from config.
    The feature allows achieving backwards compatibility with earlier gRPC client.
    """
    domain_override = "OverrideDomainName"
    with temp_config(
        {
            "runtime": {
                "library": "sample_lib",
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "domain": domain_override,
                },
            }
        }
    ) as cfg:
        # Changing the domain still effects the default package name.
        # But if you override the package, it overrides the package name completely (see next test).
        validate_package_with_override(
            ServicePackageFactory.ServiceType.INFERENCE,
            domain_override,
            f"caikit.runtime.{domain_override}",
        )
        validate_package_with_override(
            ServicePackageFactory.ServiceType.TRAINING,
            domain_override,
            f"caikit.runtime.{domain_override}",
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            cfg, "sample_lib", False
        )
        assert "SampleModule" in str(clean_modules)


def test_override_package(clean_data_model):
    """
    Test override of gRPC package generation from config.
    The feature allows achieving backwards compatibility with earlier gRPC client.
    """
    package_override = "foo.runtime.yada.v0"
    with temp_config(
        {
            "runtime": {
                "library": "sample_lib",
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "package": package_override,
                },
            }
        }
    ) as cfg:
        validate_package_with_override(
            ServicePackageFactory.ServiceType.INFERENCE, "SampleLib", package_override
        )
        validate_package_with_override(
            ServicePackageFactory.ServiceType.TRAINING, "SampleLib", package_override
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            cfg, "sample_lib", False
        )
        assert "SampleModule" in str(clean_modules)


def test_override_package_and_domain_with_proto_gen(clean_data_model):
    """
    Test override of both package and domain, to make sure they work together, and
    additionally test the proto generation.
    The feature allows achieving backwards compatibility with earlier gRPC client.
    """
    package_override = "foo.runtime.yada.v0"
    domain_override = "OverrideDomainName"
    with temp_config(
        {
            "runtime": {
                "library": "sample_lib",
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "package": package_override,
                    "domain": domain_override,
                },
            }
        }
    ) as cfg:
        inf_svc = validate_package_with_override(
            ServicePackageFactory.ServiceType.INFERENCE,
            domain_override,
            package_override,
        )
        train_svc = validate_package_with_override(
            ServicePackageFactory.ServiceType.TRAINING,
            domain_override,
            package_override,
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(
            cfg, "sample_lib", False
        )
        assert "SampleModule" in str(clean_modules)

        # Full check on proto generation
        with tempfile.TemporaryDirectory() as output_dir:
            render_dataobject_protos(output_dir)
            inf_svc.service.write_proto_file(output_dir)
            train_svc.service.write_proto_file(output_dir)

            output_dir_path = Path(output_dir)
            for proto_file in output_dir_path.glob("*.proto"):
                with open(proto_file, "rb") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip().decode("utf-8")
                        if line.startswith("package "):
                            package_name = line.split()[1]
                            assert package_name[-1] == ";"
                            package_name = package_name[0:-1]
                            root_package_name = package_name.split(".")[0]
                            if root_package_name not in {"caikit_data_model", "caikit"}:
                                assert package_name == package_override
                        elif line.startswith("service "):
                            service_name = line.split()[1]
                            domain_override_lower = domain_override.lower()
                            if proto_file.name.startswith(domain_override_lower):
                                if proto_file.stem.endswith("trainingservice"):
                                    assert (
                                        service_name
                                        == f"{domain_override}TrainingService"
                                    )
                                else:
                                    assert service_name == f"{domain_override}Service"


def test_backend_modules_included_in_service_generation(
    clean_data_model, reset_globals
):
    # Add a new backend module for the good ol' `SampleModule`
    @caikit.module(backend_type=MockBackend.backend_type, base_module=SampleModule)
    class NewBackendModule(caikit.core.ModuleBase):
        def run(
            self, sample_input: SampleInputType, backend_param: str
        ) -> SampleOutputType:
            pass

    inference_service = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE
    )
    predict_class = get_inference_request(SampleTask)
    sample_task_request = predict_class().to_proto()

    # Check that the new parameter defined in this backend module exists in the service
    assert "backend_param" in sample_task_request.DESCRIPTOR.fields_by_name.keys()


def test_get_inference_request_throws_wrong_type(runtime_grpc_server):
    with pytest.raises(TypeError) as e:
        get_inference_request(task_or_module_class="something random")
    assert "subclass check failed" in e.value.args[0]


def test_get_inference_request(runtime_grpc_server):
    """Test that we are able to get inference request DM with either module or task class"""
    assert get_inference_request(SampleModule).__name__ == "SampleTaskRequest"
    assert get_inference_request(SampleTask).__name__ == "SampleTaskRequest"
    assert (
        get_inference_request(SampleModule, output_streaming=True).__name__
        == "ServerStreamingSampleTaskRequest"
    )
    assert (
        get_inference_request(SampleTask, output_streaming=True).__name__
        == "ServerStreamingSampleTaskRequest"
    )
    assert (
        get_inference_request(
            SampleModule, input_streaming=True, output_streaming=True
        ).__name__
        == "BidiStreamingSampleTaskRequest"
    )
    assert (
        get_inference_request(
            SampleTask, input_streaming=True, output_streaming=True
        ).__name__
        == "BidiStreamingSampleTaskRequest"
    )


def test_get_train_request_throws_wrong_type(runtime_grpc_server):
    with pytest.raises(TypeError) as e:
        get_train_request("not_a_module")
    assert "subclass check failed" in e.value.args[0]


def test_get_train_request(runtime_grpc_server):
    assert (
        get_train_request(SampleModule).__name__ == "SampleTaskSampleModuleTrainRequest"
    )
    assert get_train_request(OtherModule).__name__ == "OtherTaskOtherModuleTrainRequest"


def test_get_train_params_throws_wrong_type(runtime_grpc_server):
    with pytest.raises(TypeError) as e:
        get_train_params("not_a_module")
    assert "subclass check failed" in e.value.args[0]


def test_get_train_params(runtime_grpc_server):
    assert (
        get_train_params(SampleModule).__name__
        == "SampleTaskSampleModuleTrainParameters"
    )
    assert (
        get_train_params(OtherModule).__name__ == "OtherTaskOtherModuleTrainParameters"
    )


# @pytest.mark.skip("WIP")
def test_assert_compatible_respects_previously_defined_proto_spec(
    clean_data_model_with_no_sampletaskrequest_proto,
):
    # Test that we can create API with proto fields numbering respecting backwards compatibility,
    # By first, creating a "previous version" with List Module that has 2 fields sample_input and throw
    # then, creating a "new version" with List Module and Sample Module, and the new API spec should
    # keep sample_input and throw fields, and add error as 3rd field

    protos_dir_path = os.path.join(os.getcwd(), "protos")
    # output_compiled_path = os.path.join(os.getcwd(), "compiled")
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"included": [ListModule.MODULE_ID]},
                }
            }
        },
        "merge",
    ):
        # generate a service proto spec (this will be the "previous version" with only ListModule)
        dump_grpc_services(protos_dir_path)
        # Create a "compiled" dir and then compile these protos

        with temp_module() as (mod_name, mod_dir):
            print("mod_name is: ", mod_name)
            print("mod_dir is: ", mod_dir)

            output_compiled_path = os.path.join(mod_dir, "compiled")
            os.makedirs(output_compiled_path)
            # Creates an init file
            with open(os.path.join(output_compiled_path, "__init__.py"), "w") as fp:
                pass

            # Third Party
            import pkg_resources

            grpc_protos_include = pkg_resources.resource_filename(
                "grpc_tools", "_proto"
            )

            for file in os.listdir(protos_dir_path):
                if file.endswith(".proto"):
                    proto_file = os.path.join(protos_dir_path, file)
                    protoc_args = [
                        "grpc_tools.protoc",
                        "--proto_path=" + protos_dir_path,
                        "--proto_path=" + grpc_protos_include,
                        "--grpc_python_out=" + output_compiled_path,
                        "--python_out=" + output_compiled_path,
                        proto_file,
                    ]
                try:
                    retval = protoc.main(protoc_args)
                except Exception as e:  # pragma: no cover
                    raise RuntimeError(
                        "Failed to compile proto files {}".format(repr(e))
                    )
                if retval != 0:
                    raise RuntimeError(
                        "Proto compilation returned code {}".format(retval)
                    )
            print("compiled has ", os.listdir(output_compiled_path))

            # import this compiled service
            # Standard
            import importlib
            import sys

            sys.path.append(output_compiled_path)

            # samplelib_pb2 = __import__("samplelibservice_pb2", fromlist=[mod_name])

    #         print("dpool in test is: ", dpool)
    #         ApiFieldNames.add_proto_spec(samplelibservice_pb2, d_pool=dpool)
    #         # these fields for SampleTaskRequest comes from ListModule
    #         assert ApiFieldNames.get_fields_for_message("SampleTaskRequest") == {
    #             "sample_input": 1,
    #             "throw": 2,
    #         }

    # # call service generation, and generate another service with ListModule and SampleModule
    # with temp_config(
    #     {
    #         "runtime": {
    #             "service_generation": {
    #                 "module_guids": {
    #                     "included": [ListModule.MODULE_ID, SampleModule.MODULE_ID]
    #                 },
    #             }
    #         }
    #     },
    #     "merge",
    # ):
    #     with temp_dpool(
    #         inherit_global=True, skip_inherit=[".*sampletaskrequest.proto"]
    #     ) as dpool:
    #         service_pckg = ServicePackageFactory.get_service_package(
    #             ServicePackageFactory.ServiceType.INFERENCE
    #         )
    #         sample_task_request_fields = (
    #             service_pckg.messages.SampleTaskRequest.DESCRIPTOR.fields
    #         )
    #         fields = {field.name: field.number for field in sample_task_request_fields}
    #         # assert that the fields from previous version still exist
    #         assert ("sample_input", 1) in fields.items()
    #         assert ("throw", 2) in fields.items()
    #         # assert that the new SampleTaskRequest now will have an extra field (error, 3) that comes from SampleModule
    #         assert ("error", 3) in fields.items()

    # Clean up
    shutil.rmtree(protos_dir_path, ignore_errors=True)
