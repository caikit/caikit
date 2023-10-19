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
A sample module for sample things!
"""
# Standard
from typing import List

# Local
from ...data_model.sample import PrimitiveTask
from caikit.core.modules import ModuleLoader
from caikit.interfaces.common.data_model import StrSequence
import caikit.core


@caikit.core.module(
    "bd87fbbe-caad-4db6-93f5-37cc3dbedef5", "PrimitiveModule", "0.0.1", PrimitiveTask
)
class PrimitiveModule(caikit.core.ModuleBase):
    def run(
        self,
        str_type: str,
        list_str_type: List[str],
        int_type: int,
        list_int_type: List[int],
    ) -> StrSequence:
        return StrSequence(
            values=[
                str_type,
                *list_str_type,
                str(int_type),
                *[str(i) for i in list_int_type],
            ]
        )

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls()
