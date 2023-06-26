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


"""Augmentor representing multiple augmentor behaviors combined into one configurable class.
"""
# First Party
import alog

# Local
from ..toolkit.errors import error_handler
from .base import AugmentorBase
from .schemes import SchemeBase

log = alog.use_channel("MRGD_AUGMNTR")
error = error_handler.get(log)


class MergedAugmentor(AugmentorBase):
    def __init__(self, scheme):
        """Build an augmentor encapsulating multiple augmentors, where application order is
        governed by the provided scheme.
        Args:
            scheme (SchemeBase): Scheme indicating how encapsulated augmentors
                should be combined.
        """
        # NOTE: Random seed of merged augmentor does not currently matter since randomness is
        # already represented completely within encapsulated augmentors and scheme state
        super().__init__(random_seed=1001)
        error.type_check("<COR10421239E>", SchemeBase, scheme=scheme)
        self._scheme = scheme
        # pylint: disable=consider-using-set-comprehension
        augmentor_types = set([aug.augmentor_type for aug in self._scheme._augmentors])
        error.value_check(
            "<COR18849029E>",
            len(augmentor_types) == 1,
            "Cannot merge augmentors with differing augmentor types",
        )
        error.value_check(
            "<COR18146529E>",
            all(not aug.produces_none for aug in self._scheme._augmentors),
            "Cannot merge augmentors that produces <None> outputs",
        )
        self.augmentor_type = augmentor_types.pop()

    def _augment(self, obj):
        """Apply a merged augmentor whose behavior is controlled by the encapsulated scheme.

        Args:
            obj (str | caikit.core.data_model.DataBase): Object to be augmented.
        Returns:
            str | caikit.core.data_model.DataBase: Augmented object of same type
                as input obj.
        """
        return self._scheme.execute(obj)

    def reset(self):
        """Reset the merge augmentor's random number generation. In this case we actually don't
        need to care about the augmentor random state; what matters for this is the augmentor.
        scheme.
        """
        self._scheme.reset()
