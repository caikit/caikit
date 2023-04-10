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
import random

# Third Party
import pytest

# Local
from caikit.core.augmentors.base import AugmentorBase

# Unit Test Infrastructure
from tests.base import TestCaseBase


class StubAugmentor(AugmentorBase):
    """Test class for augmentor core."""

    augmentor_type = str
    # Indicates that this class should not be picked up by augmentor discovery utilities.
    is_test_augmentor = True

    def __init__(self):
        super().__init__(random_seed=1001)

    def _augment(self, obj):
        """Takes a string and returns 'happy', unless the string 'sad' is provided, in which case
        we return an incorrect type.
        """
        if obj == "sad":
            return 1
        else:
            return "happy"


class TestAugmentorBase(TestCaseBase):
    def setUp(self):
        self.test_augmentor = StubAugmentor()

    def test_augment_input_type_enforcement(self):
        """Ensure that input augmentor type is enforced."""
        with self.assertRaises(TypeError):
            self.test_augmentor.augment(True)

    def test_augment_output_type_enforcement(self):
        """Ensure that output augmentor type is enforced."""
        with self.assertRaises(TypeError):
            self.test_augmentor.augment("sad")

    def test_augment_happy_path(self):
        """Ensure that if we pass a happy string, we get a string back."""
        aug_out = self.test_augmentor.augment("happy")
        self.assertIsInstance(aug_out, str)
        self.assertEqual(aug_out, "happy")

    def test_reset_rewinds_random_state(self):
        """Test that if we reset the augmentor, our random state is rewound."""
        # Generate the first 6 numbers using the internal random state
        first_three = [random.random() for x in range(3)]
        next_three = [random.random() for x in range(3)]
        # Make sure the first 3 don't match the second 3
        self.assertNotEqual(first_three, next_three)
        # Then reset and ensure we can regenerate the first 3
        self.test_augmentor.reset()
        reset_three = [random.random() for x in range(3)]
        self.assertEqual(first_three, reset_three)
        # Sanity check to make sure we aren't just getting the same thing both times
        self.assertNotEqual(first_three, next_three)
