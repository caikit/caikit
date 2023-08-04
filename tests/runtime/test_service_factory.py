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
from typing import Tuple
import tempfile

# Third Party
from google.protobuf.message import Message

# Local
from caikit.core.data_model import render_dataobject_protos
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from sample_lib.modules.sample_task import ListModule
from tests.conftest import temp_config
import caikit

## Helpers #####################################################################


def validate_infer_train_with_override(
    domain_override: str, package_override: str
) -> Tuple[ServicePackage, ServicePackage]:
    """Construct and validate train and infer service packages with domain and package path override

    The config should be set up for domain and/or package override, before this function is called.

    Args:
        domain_override (str): The domain to validate (maybe overridden value, or the default)
        package_override (str): The package path to validate (maybe overridden value, or the default)

    Returns: Tuple of created train and infer service packages, for potential subsequent reuse
    """

    """Compare the string representation

    Args:
        obj: Any
            Object to compare against the contents of the file
        file_path: str
            Location of the file containing the expected string value of `obj`
    """

    inf_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
    )
    train_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
    )
    inf_service_name = f"{domain_override}Service"
    assert inf_svc.service.__name__ == inf_service_name
    assert inf_svc.descriptor.full_name == f"{package_override}.{inf_service_name}"
    for message_name in [
        x for x in inf_svc.messages.__dict__.keys() if not x.startswith("_")
    ]:
        message: Message = getattr(inf_svc.messages, message_name)
        assert message.DESCRIPTOR.full_name == f"{package_override}.{message_name}"

    train_svc_name = f"{domain_override}TrainingService"
    assert train_svc.service.__name__ == train_svc_name
    assert train_svc.descriptor.full_name == f"{package_override}.{train_svc_name}"
    for message_name in [
        x for x in train_svc.messages.__dict__.keys() if not x.startswith("_")
    ]:
        message: Message = getattr(train_svc.messages, message_name)
        assert message.DESCRIPTOR.full_name == f"{package_override}.{message_name}"
    return inf_svc, train_svc


### Private method tests #############################################################


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
                "service_generation": {"task_types": {"excluded": ["SampleTask"]}}
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" not in str(clean_modules)


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
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "ListModule" not in str(clean_modules)
        assert "SampleModule" in str(clean_modules)
        assert "OtherModule" in str(clean_modules)


def test_get_and_filter_modules_respects_excluded_modules_and_excluded_task_type():
    assert "InnerModule" in str(MODULE_LIST)
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"excluded": [ListModule.MODULE_ID]},
                    "task_types": {"excluded": ["OtherTask"]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "ListModule" not in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)
        assert "SampleModule" in str(clean_modules)


def test_get_and_filter_modules_respects_included_modules_and_included_task_types():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "module_guids": {"included": [ListModule.MODULE_ID]},
                    "task_types": {"included": ["OtherTask"]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert len(clean_modules) == 2
        assert "OtherModule" in str(clean_modules)
        assert "ListModule" in str(clean_modules)


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
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert len(clean_modules) == 1
        assert "ListModule" in str(clean_modules)
        assert "SampleModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_task_types():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)
        assert "OtherModule" not in str(clean_modules)
        # InnerModule has no task
        assert "InnerModule" not in str(clean_modules)


def test_get_and_filter_modules_respects_included_task_types_and_excluded_modules():
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "task_types": {"included": ["SampleTask"]},
                    "module_guids": {"excluded": [ListModule.MODULE_ID]},
                }
            }
        }
    ) as cfg:
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)
        assert "ListModule" not in str(clean_modules)


def test_override_domain():
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
        validate_infer_train_with_override(
            domain_override, f"caikit.runtime.{domain_override}"
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)


def test_override_package():
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
        validate_infer_train_with_override("SampleLib", package_override)

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)


def test_override_package_and_domain_with_proto_gen():
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
        inf_svc, train_svc = validate_infer_train_with_override(
            domain_override, package_override
        )

        # Just double-check that basics are good.
        clean_modules = ServicePackageFactory._get_and_filter_modules(cfg, "sample_lib")
        assert "SampleModule" in str(clean_modules)

        # Full check on proto generation
        with tempfile.TemporaryDirectory() as output_dir:
            render_dataobject_protos(output_dir)
            inf_svc.service.write_proto_file(output_dir)
            train_svc.service.write_proto_file(output_dir)

            output_dir_path = Path(output_dir)
            for proto_file in output_dir_path.glob("*.proto"):
                # print(proto_file)
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
