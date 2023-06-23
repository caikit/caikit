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


"""Core base class for Augmentor combination schemes.
"""
# Standard
import random

# First Party
import alog

# Local
from ...toolkit.errors import error_handler
from .. import AugmentorBase

log = alog.use_channel("AUG_SCHEME_BASE")
error = error_handler.get(log)


class SchemeBase:
    def __init__(self, preserve_order, augmentors, random_seed):
        """Initialize the core components of a merging scheme to be leveraged when combining
        augmentors.

        Args:
            preserve_order (bool): Indicates whether or not the contained
                augmentors should always be considered in the order that they
                were provided when they are being applied.
            augmentors (list(AugmentorBase) | tuple(AugmentorBase)): List or
                tuple of Augmentor objects to be applied.
            random_seed (int): Random seed for controlling shuffling behavior.
        """
        error.type_check("<COR54555981E>", bool, preserve_order=preserve_order)
        error.type_check("<COR54155111E>", list, tuple, augmentors=augmentors)
        error.type_check("<COR73170110E>", int, random_seed=random_seed)
        error.value_check(
            "<COR67355718E>",
            len(augmentors) > 0,
            "Must provide at least one augmentor to build a scheme.",
        )
        error.type_check_all("<COR37249765E>", AugmentorBase, augmentors=augmentors)
        # Determine whether or not augmentors should be applied in the order provided or
        # applied in random order.
        self._preserve_order = preserve_order
        self._current_order = list(range(len(augmentors)))
        self._augmentors = augmentors
        self._init_state = random.getstate()

    def execute(self, obj):
        """Execute the merged scheme, i.e., augment the object by leveraging the encapsulated
        augmentors.

        Args:
            obj (str | caikit.core.data_model.DataBase): Object to be augmented.
        Returns:
            str | caikit.core.data_model.DataBase: Augmented object of same type
                as input obj.
        """
        if not self._preserve_order:
            random.shuffle(self._current_order)
        return self._execute(obj)

    def reset(self):
        """Reset the random state of all encapsulated augmentors and the scheme itself."""
        # Reset the random state for all augmentors
        for aug in self._augmentors:
            aug.reset()
        # Reset the random state for the scheme using the default random package
        random.setstate(self._init_state)
