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

# Standard
# We're filtering (most) warnings for now
import warnings as _warnings

_warnings.filterwarnings("ignore")

# Local
# must import toolkit first since we need alog to be set up before it is used
from . import (
    blocks,
    config,
    data_model,
    module,
    module_config,
    resources,
    toolkit,
    workflows,
)
from .blocks.base import BlockBase, block
from .config import *
from .data_model import dataobject
from .model_manager import *
from .module import *
from .module_backend_config import configure as backend_configure
from .module_backends import *
from .module_config import ModuleConfig
from .resources.base import ResourceBase, resource
from .toolkit import *
from .workflows.base import WorkflowBase, workflow

# Configure the global model wrangling functions
MODEL_MANAGER = ModelManager()
extract = MODEL_MANAGER.extract
load = MODEL_MANAGER.load
resolve_and_load = MODEL_MANAGER.resolve_and_load
