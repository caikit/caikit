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

# Third Party
import pytest

# Local
from caikit.runtime.service_factory import ServicePackageFactory
from sample_lib.modules.sample_task import ListModule
from tests.conftest import temp_config
import caikit

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
