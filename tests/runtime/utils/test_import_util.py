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
from types import SimpleNamespace
import inspect
import sys

# Third Party
import pytest

# Local
from caikit.core.data_model.base import DataBase
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import (
    UnifiedDataModel,
    clean_lib_names,
    get_data_model,
    get_dynamic_module,
)

## Setup #########################################################################

multi_lib_names = "caikit_libraryfoo[foo.bar]>=1.0.0-rc15 caikit_librarybar foo-bar==1"
multi_lib_cleaned_list = ["caikit_libraryfoo", "caikit_librarybar", "foo_bar"]

## Tests ########################################################################


### clean_lib_names #############################################################
def test_multi_lib_name_cleaning():
    name_list = clean_lib_names(multi_lib_names)
    assert len(name_list) == len(multi_lib_cleaned_list)
    assert multi_lib_cleaned_list == name_list


def test_single_lib_name_cleaning():
    name_list = clean_lib_names("caikit_libraryfoo[foo.bar]>=1.0.0-rc15")
    assert name_list[0] == "caikit_libraryfoo"


def test_single_lib_name_no_option_dep():
    name_list = clean_lib_names("caikit_libraryfoo>=1.0.0-rc15")
    assert name_list[0] == "caikit_libraryfoo"


def test_single_lib_name_no_version():
    name_list = clean_lib_names("caikit_libraryfoo[foo.bar]")
    assert name_list[0] == "caikit_libraryfoo"


### get_dynamic_module #############################################################


def test_get_caikit_library_throws_on_nonimportable_lib():
    """If an invalid module is provided to caikit lib setup, it throws a ValueError"""
    with pytest.raises(CaikitRuntimeException):
        get_dynamic_module("caikit_bad_lib")


def test_get_caikit_library_loads_sample_stdlib_module():
    """If a valid module is provided to caikit lib setup, that module is returned"""
    # Standard
    import sys

    sample_module = get_dynamic_module("sys")
    assert sample_module == sys


def test_get_caikit_library_loads_caikit_core():
    """If caikit.core is provided to caikit lib setup, the module is returned"""
    # Local
    import caikit.core

    sample_module = get_dynamic_module("caikit.core")
    assert sample_module == caikit.core


### get_data_model #############################################################


def test_get_data_model_throws_on_nonimportable_lib():
    """If an invalid module is provided to get_data_model, it throws a ValueError"""
    mock_config = SimpleNamespace(**{"caikit_library": "caikit_bad_lib"})
    with pytest.raises(CaikitRuntimeException):
        get_data_model(mock_config)


def test_get_data_model_ok_on_lib_with_no_data_model():
    """If a valid module with no data model is provided to get_data_model it
    returns an empty module object
    """
    mock_config = SimpleNamespace(**{"caikit_library": "sys"})
    data_model = get_data_model(mock_config)
    assert isinstance(data_model, UnifiedDataModel)


def test_get_data_model_is_accessible():
    """get_data_model should return the stuff in the `sample_lib.data_model` package"""
    # Third Party
    import sample_lib

    cdm = get_data_model()
    attrs_match = [
        (hasattr(cdm, attr_name) and getattr(cdm, attr_name) == attr_val)
        for attr_name, attr_val in vars(sample_lib.data_model).items()
        if not attr_name.startswith("_")
        and inspect.isclass(attr_val)
        and issubclass(attr_val, DataBase)
    ]
    assert all(attrs_match)


def test_multiple_caikit_libraries():
    """Make sure that multiple libraries can be imported and merged into a
    unified data model
    """
    lib_names = ["caikit.interfaces.runtime", "sample_lib"]
    # sys.path.append(
    #     os.path.realpath(
    #         os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
    #     )
    # )
    mock_config = SimpleNamespace(**{"caikit_library": " ".join(lib_names)})
    cdm = get_data_model(mock_config)
    for lib_name in lib_names:
        assert lib_name in sys.modules
        lib_mod = sys.modules[lib_name]
        attrs_match = [
            (hasattr(cdm, attr_name) and getattr(cdm, attr_name) == attr_val)
            for attr_name, attr_val in vars(lib_mod.data_model).items()
            if not attr_name.startswith("_")
            and inspect.isclass(attr_val)
            and issubclass(attr_val, DataBase)
        ]
        assert all(attrs_match)
