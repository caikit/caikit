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


"""Scheme for randomly picking 1 augmentor from a list per object.
"""
# Standard
import random

# First Party
import alog

# Local
from ...toolkit.errors import error_handler
from .base import SchemeBase

log = alog.use_channel("RSING_AUG_SCHEME")
error = error_handler.get(log)


class RandomSingleSelectionScheme(SchemeBase):
    def __init__(self, selection_probs, augmentors, random_seed=1001):
        """Create a merging augmentor scheme which randomly applies one of many encapsulated
        augmentors when executed.

        Args:
            selection_probs (list(int|float) | tuple(int|float)): Probability
                values for applying each augmentor (must sum to 1).
            augmentors (list(AugmentorBase) | tuple(AugmentorBase)): Augmentors
                to be applied (in same order as selection_probs).
            random_seed (int): Random seed for controlling shuffling behavior.
        """
        super().__init__(True, augmentors, random_seed)
        error.type_check("<COR26721310E>", list, tuple, selection_probs=selection_probs)
        error.type_check_all(
            "<COR89551931E>", int, float, selection_probs=selection_probs
        )
        error.value_check(
            "<COR22821754E>",
            len(selection_probs) == len(augmentors),
            "Number of selection probabilties must match the number of augmentors",
        )
        error.value_check(
            "<COR82891072E>",
            all(0 <= prob <= 1 for prob in selection_probs),
            "Selection probabilities must be in the range [0, 1]",
        )
        error.value_check(
            "<COR00872610E>",
            sum(selection_probs) == 1,
            "Selection probabilities must sum up to one to create a single selection scheme",
        )
        self._selection_probs = selection_probs

    def _execute(self, obj):
        """Execute the merged scheme by picking one random augmentor and applying it.

        Args:
            obj (str | caikit.core.data_model.DataBase): Object to be augmented.
        Returns:
            str | caikit.core.data_model.DataBase: Augmented object of same type
                as input obj.
        """
        aug = random.choices(self._augmentors, weights=self._selection_probs)[0]
        return aug.augment(obj)
