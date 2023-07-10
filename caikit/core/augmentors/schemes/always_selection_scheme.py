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


"""Scheme for always selecting an augmentation behavior.
"""
# Local
from .base import SchemeBase


class AlwaysSelectionScheme(SchemeBase):
    def __init__(self, preserve_order, augmentors, random_seed=1001):
        """Create a merging augmentor scheme which always applies every contained augmentor.

        Args:
            preserve_order (bool): Indicates whether or not the contained
                augmentors should always be considered in the order that they
                were provided when they are being applied.
            augmentors (list(AugmentorBase) | tuple(AugmentorBase)): Augmentors
                to be applied (in same order as selection_probs).
            random_seed (int): Random seed for controlling shuffling behavior.
        """
        super().__init__(preserve_order, augmentors, random_seed)

    def _execute(self, obj):
        """Execute the merged scheme by always applying every contained augmentor (in a potentially
        shuffled ordering, based on the value of self.preserve_order).

        Args:
            obj (str | caikit.core.data_model.DataBase): Object to be augmented.
        Returns:
            str | caikit.core.data_model.DataBase: Augmented object of same type
                as input obj.
        """
        output_obj = obj
        for idx in self._current_order:
            aug = self._augmentors[idx]
            output_obj = aug.augment(output_obj)
        return output_obj
