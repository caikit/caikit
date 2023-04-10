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
from typing import Type

# First Party
import alog

# Local
from caikit.core import ModuleBase

log = alog.use_channel("LOADED-MODEL")


class LoadedModel:
    class Builder:
        def __init__(self):
            self._model_to_build = LoadedModel()

        def module(self, caikit_module: ModuleBase) -> "LoadedModel.Builder":
            self._model_to_build._caikit_module = caikit_module
            return self

        def path(self, model_path: str) -> "LoadedModel.Builder":
            self._model_to_build._model_path = model_path
            return self

        def type(self, model_type: str) -> "LoadedModel.Builder":
            self._model_to_build._model_type = model_type
            return self

        def id(self, model_id: str) -> "LoadedModel.Builder":
            self._model_to_build._model_id = model_id
            return self

        def build(self) -> "LoadedModel":
            return self._model_to_build

    def __init__(self):
        # Use the builder ^^
        self._caikit_module = None
        self._model_id = ""
        self._model_path = ""
        self._model_type = ""
        self._size = None

    def id(self) -> str:
        return self._model_id

    def module(self) -> Type[ModuleBase]:
        return self._caikit_module

    def type(self) -> str:
        return self._model_type

    def path(self) -> str:
        return self._model_path

    def size(self) -> int:
        if self._size is None:
            return 0
        return self._size

    def has_size(self) -> bool:
        return self._size is not None and self._size > 0

    # Size is the only mutable member, but only mutable once
    def set_size(self, model_size: int):
        if self._size is None or self._size == 0:
            self._size = model_size
        elif self._size != model_size:
            log.warning(
                "COM62206343W",
                "Attempted to set size of model %s to %v, but it was already %v",
                self.id(),
                model_size,
                self.size(),
            )
