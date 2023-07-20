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
A LoadedModel is a metadata wrapper around an instance of a core.ModuleBase
class that contains the additional information needed to manage that model in
the runtime.
"""

# Standard
from concurrent.futures import Future
from typing import Callable, Optional

# First Party
import alog

# Local
from caikit.core import ModuleBase
from caikit.core.toolkit.errors import error_handler
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("LOADED-MODEL")
error = error_handler.get(log)


# A future object that will yield an instance of a caikit module (a model)
# NOTE: 3.9 introduced subscript typing for Futures
try:
    CaikitModelFuture = Future[ModuleBase]  # pylint: disable=unsubscriptable-object
except TypeError:  # pragma: no cover
    CaikitModelFuture = Future


class LoadedModel:
    __doc__ = __doc__

    class Builder:
        """The LoadedModel.Builder allows the LoadedModel instance to be
        constructed in pieces with chained '.' getattr semantics.
        """

        def __init__(self):
            self._model_to_build = LoadedModel()

        def model_future(
            self, caikit_model_future: CaikitModelFuture
        ) -> "LoadedModel.Builder":
            self._model_to_build._caikit_model_future = caikit_model_future
            return self

        def fail_callback(self, callback: Callable) -> "LoadedModel.Builder":
            self._model_to_build._fail_callback = callback
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
            error.value_check(
                "<RUN12786023E>",
                self._model_to_build._caikit_model_future
                and self._model_to_build._model_id
                and self._model_to_build._model_type,
                "Cannot build LoadedModel with incomplete required fields."
                + " Future: {}, ID: {}, Type: {}",
                self._model_to_build._caikit_model_future,
                self._model_to_build._model_id,
                self._model_to_build._model_type,
            )
            return self._model_to_build

    def __init__(self):
        # Use the builder ^^
        self._caikit_model_future: Optional[CaikitModelFuture] = None
        self._model: Optional[ModuleBase] = None
        self._fail_callback: Optional[Callable] = None
        self._model_id: str = ""
        self._model_path: str = ""
        self._model_type: str = ""
        self._size: Optional[int] = None

    def id(self) -> str:
        return self._model_id

    def model(self) -> ModuleBase:
        self.wait()
        return self._model

    def wait(self):
        if self._model is None:
            try:
                self._model = self._caikit_model_future.result()
            except CaikitRuntimeException:
                if self._fail_callback:
                    self._fail_callback()
                raise

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
                "<RUN46815705W>",
                "Attempted to set size of model %s to %s, but it was already %s",
                self.id(),
                model_size,
                self.size(),
            )
