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

# Third Party
import pytest

# Local
from caikit.core.augmentors import AugmentorBase, MergedAugmentor, schemes

# Unit Test Infrastructure
from tests.base import TestCaseBase


class StubStringAugmentor(AugmentorBase):
    augmentor_type = str
    # Indicates that this class should not be picked up by augmentor discovery utilities.
    is_test_augmentor = True

    def __init__(self, identifier):
        super().__init__(random_seed=1001)
        self.id = identifier

    def _augment(self, obj):
        return "{} {}".format(obj, self.id)


class StubIntegerAugmentor(AugmentorBase):
    augmentor_type = int
    # Indicates that this class should not be picked up by augmentor discovery utilities.
    is_test_augmentor = True

    def __init__(self, identifier):
        super().__init__(random_seed=1001)
        self.id = identifier

    def _augment(self, obj):
        return obj


class StubNoneProducingAugmentor(AugmentorBase):
    """Test class augmentor schemes / merge augmentors."""

    augmentor_type = int
    # Indicates that this class should not be picked up by augmentor discovery utilities.
    is_test_augmentor = True

    def __init__(self):
        super().__init__(random_seed=1001, produces_none=True)


class TestAugmentorSchemes(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.augmentors = [StubStringAugmentor("first"), StubStringAugmentor("second")]

    ###############################################################################################
    #                        Happy path cases for all Augmentor Schemes                           #
    ###############################################################################################
    # Note: For the tests below, since we call _execute(), not execute(), preserve order
    # value does not matter in these tests. execute() is a thin wrapper around ._execute()
    # that has an alternate path for random shuffling.
    def test_always_selection_scheme(self):
        """Ensure that an "always" selection scheme applies all augmentors."""
        always_scheme = schemes.AlwaysSelectionScheme(True, self.augmentors)
        self.assertEqual(always_scheme._execute("text"), "text first second")

    def test_multi_selection_scheme(self):
        """Random state independent test cases for random multi selection scheme."""
        always_multi_scheme = schemes.RandomMultiSelectionScheme(
            True, [1, 1], self.augmentors
        )
        never_multi_scheme = schemes.RandomMultiSelectionScheme(
            True, [0, 0], self.augmentors
        )
        self.assertEqual(always_multi_scheme._execute("text"), "text first second")
        self.assertEqual(never_multi_scheme._execute("text"), "text")

    def test_single_selection_scheme(self):
        """Random state independent test cases for random single selection scheme."""
        first_scheme = schemes.RandomSingleSelectionScheme([1, 0], self.augmentors)
        second_scheme = schemes.RandomSingleSelectionScheme([0, 1], self.augmentors)
        self.assertEqual(first_scheme._execute("text"), "text first")
        self.assertEqual(second_scheme._execute("text"), "text second")

    ###############################################################################################
    #                   Input validation for Augmentor Scheme initialization                      #
    ###############################################################################################
    # Base class
    def test_base_checks_preserve_order_type(self):
        """Ensure that preserve order needs to be a boolean value."""
        self.assertRaises(TypeError, schemes.SchemeBase, None, self.augmentors, 1001)

    def test_base_validates_augmentor_types(self):
        """Ensure SchemeBase type checks augmentor list/tuple properly."""
        self.assertRaises(TypeError, schemes.SchemeBase, True, "Foobar", 1001)
        self.assertRaises(TypeError, schemes.SchemeBase, True, ["Foobar"], 1001)

    def test_base_requires_augmentors(self):
        """Ensure SchemeBase requires at least one augmentor."""
        self.assertRaises(ValueError, schemes.SchemeBase, True, [], 1001)

    # Random Multi Selection Scheme
    def test_multi_selection_prob_types(self):
        """Ensure that selection probability types are handled correctly."""
        self.assertRaises(
            TypeError,
            schemes.RandomMultiSelectionScheme,
            True,
            ["a", "b"],
            self.augmentors,
            1001,
        )
        self.assertRaises(
            TypeError,
            schemes.RandomMultiSelectionScheme,
            True,
            "foo",
            self.augmentors,
            1001,
        )

    def test_multi_selection_prob_length_alignment(self):
        """Ensure that the number of probability values needs to match the number of augmentors."""
        self.assertRaises(
            ValueError,
            schemes.RandomMultiSelectionScheme,
            True,
            [1],
            self.augmentors,
            1001,
        )
        self.assertRaises(
            ValueError,
            schemes.RandomMultiSelectionScheme,
            True,
            [1, 2, 3],
            self.augmentors,
            1001,
        )

    def test_multi_selection_prob_range(self):
        """Ensure that selection probability ranges are properly validated."""
        self.assertRaises(
            ValueError,
            schemes.RandomMultiSelectionScheme,
            True,
            [-1, 2],
            self.augmentors,
            1001,
        )

    # Random Single Selection Scheme
    def test_single_selection_prob_types(self):
        """Ensure that selection probability types are handled correctly."""
        self.assertRaises(
            TypeError,
            schemes.RandomSingleSelectionScheme,
            ["a", "b"],
            self.augmentors,
            1001,
        )
        self.assertRaises(
            TypeError, schemes.RandomSingleSelectionScheme, "foo", self.augmentors, 1001
        )

    def test_single_selection_prob_length_alignment(self):
        """Ensure that the number of probability values needs to match the number of augmentors."""
        self.assertRaises(
            ValueError, schemes.RandomSingleSelectionScheme, [1], self.augmentors, 1001
        )
        self.assertRaises(
            ValueError,
            schemes.RandomSingleSelectionScheme,
            [1, 2, 3],
            self.augmentors,
            1001,
        )

    def test_single_selection_prob_sum_must_be_one(self):
        """Ensure that the selection probability sum of a select one problem must be one."""
        self.assertRaises(
            ValueError,
            schemes.RandomSingleSelectionScheme,
            [0.2, 0.4],
            self.augmentors,
            1001,
        )

    def test_single_selection_prob_range(self):
        """Ensure that selection probability ranges are properly validated."""
        self.assertRaises(
            ValueError,
            schemes.RandomSingleSelectionScheme,
            [-1, 2],
            self.augmentors,
            1001,
        )


class TestMergedAugmentor(TestCaseBase):
    def test_incompatible_augmentor_type_initialization(self):
        """Ensure that we raise if incompatible augmentors are provided."""
        str_aug = StubStringAugmentor("text")
        int_aug = StubIntegerAugmentor(13)
        scheme = schemes.RandomSingleSelectionScheme([1, 0], [str_aug, int_aug])
        self.assertRaises(ValueError, MergedAugmentor, scheme)

    def test_compatible_augmentor_type_initialization(self):
        """Ensure that we can build a Merged Augmentor with a well-formed merging scheme."""
        str_aug = StubStringAugmentor("text")
        scheme = schemes.RandomSingleSelectionScheme([1, 0], [str_aug, str_aug])
        aug = MergedAugmentor(scheme)
        self.assertEqual(aug.augmentor_type, str)

    def test_scheme_type_validation(self):
        """Ensure that MergedAugmentor raises if a bad arg type is provided for scheme."""
        self.assertRaises(TypeError, MergedAugmentor, 13)

    def test_merge_with_produces_none_fails(self):
        """Ensure that we cannot merge augmentors if any of them can produce <None>."""
        int_aug = StubIntegerAugmentor(13)
        none_int_aug = StubNoneProducingAugmentor()
        scheme = schemes.RandomSingleSelectionScheme([1, 0], [int_aug, none_int_aug])
        self.assertRaises(ValueError, MergedAugmentor, scheme)
