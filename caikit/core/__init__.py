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


"""Caikit Core AI Framework library.  This is the base framework for core AI/ML libraries.
"""

# the import order cannot adhere to the linter here because we must do things like
# disable warnings, initialize the JVM and configure logging in a specific order
# pylint: disable=wrong-import-position,wrong-import-order

# NOTE: There are cyclic imports due to the "import *"s here, when modules then
# "import core"

# Standard
# We're filtering (most) warnings for now
import warnings as _warnings

_warnings.filterwarnings("ignore")

# Local
from .data_model import DataObjectBase, dataobject
from .model_manager import *
from .module_backends import *
from .modules import ModuleBase, ModuleConfig, ModuleLoader, ModuleSaver, module
from .task import TaskBase, task
from .toolkit import *

# Configure the global model wrangling functions
MODEL_MANAGER = ModelManager()
extract = MODEL_MANAGER.extract
load = MODEL_MANAGER.load
resolve_and_load = MODEL_MANAGER.resolve_and_load
train = MODEL_MANAGER.train
get_model_future = MODEL_MANAGER.get_model_future
