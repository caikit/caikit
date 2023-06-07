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
import json
import os
import pickle

# Local
from caikit.core import data_model as core_dm
from caikit.core.augmentors import AugmentorBase
from sample_lib.data_model.sample import SampleInputType, SampleOutputType
from sample_lib.modules.sample_task.sample_implementation import SampleModule

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core


def build_test_augmentor(produces_none):
    class TestStreamAugmentor(AugmentorBase):
        """A 'normal' augmentor, which when leveraged correctly, always produces an output per input."""

        augmentor_type = int
        # Indicates that this class should not be picked up by augmentor discovery utilities.
        is_test_augmentor = True

        def __init__(self):
            super().__init__(random_seed=1001, produces_none=produces_none)

        def _augment(self, obj):
            """If self.produces_none is True, filters out falsy values, i.e., 0s, otherwise adds
            five to the original input.

            Also return None if we pass 13 and set self.produces_none=False, as a sad test case.
            This should raise a TypeError within the augmentor base class.
            """
            if self.produces_none and not obj:
                return None
            if not self.produces_none and obj == 13:
                return None
            return obj + 5

    return TestStreamAugmentor()


def test_data_stream_from_jsonl_is_pickleable(tmp_path):
    tmpdir = str(tmp_path)

    data = [1, 2, 3, 4, 5, 6]
    filepath = os.path.join(tmpdir, "foo.jsonl")
    with open(filepath, "w") as f:
        json.dump(data, f)

    stream = core_dm.DataStream.from_jsonl(filepath)

    pre_pickle_vals = list(stream)
    pickled_stream = pickle.loads(pickle.dumps(stream))
    post_pickle_vals = list(pickled_stream)

    assert pre_pickle_vals == post_pickle_vals


class TestDataStream(TestCaseBase):
    def setUp(self):
        self.list_data_stream = core_dm.DataStream.from_iterable(
            ["hello", "world", "!"]
        )

        self.samples_path = os.path.join(self.fixtures_dir, "data_stream_inputs")
        self.txt_data_stream = core_dm.DataStream.from_txt(
            os.path.join(self.samples_path, "sample.txt")
        )
        self.txt_collection_data_stream = core_dm.DataStream.from_txt_collection(
            os.path.join(self.samples_path, "sample_txt_collection")
        )
        self.json_collection_data_stream = core_dm.DataStream.from_json_collection(
            os.path.join(self.samples_path, "sample_json_collection")
        )
        self.csv_collection_data_stream = core_dm.DataStream.from_csv_collection(
            os.path.join(self.samples_path, "sample_csv_collection")
        )
        self.jsonl_collection_data_stream = core_dm.DataStream.from_jsonl_collection(
            os.path.join(self.samples_path, "sample_jsonl_collection")
        )
        self.csv_data_stream = core_dm.DataStream.from_csv(
            os.path.join(self.samples_path, "sample.csv")
        )
        self.csv_header_data_stream = core_dm.DataStream.from_header_csv(
            os.path.join(self.samples_path, "sample_w_header.csv")
        )
        self.json_data_stream = core_dm.DataStream.from_json_array(
            os.path.join(self.samples_path, "sample.json")
        )
        self.jsonl_data_stream = core_dm.DataStream.from_jsonl(
            os.path.join(self.samples_path, "sample.jsonl")
        )
        self.jsonl_w_control_chars_data_stream = core_dm.DataStream.from_jsonl(
            os.path.join(self.samples_path, "control_chars.jsonl")
        )
        self.sample_type_data_stream = self.txt_data_stream.map(
            lambda text: SampleInputType(name=text)
        )
        self.bad_json = os.path.join(self.samples_path, "bad_file.json")

    ## helper functions ###

    def validate_data_stream(
        self, data_stream, length, data_item_type, data_item_length=None
    ):
        self.assertIsInstance(data_stream, core_dm.DataStream)
        self.assertEqual(len(data_stream), length)
        self.assertEqual(sum(1 for _ in data_stream), length)

        for data_item in data_stream:
            self.assertIsInstance(data_item, data_item_type)
            if data_item_length is not None:
                self.assertEqual(len(data_item), data_item_length)

    ## verify the lengths and types of our fixture data streams

    def test_list_data_stream(self):
        self.validate_data_stream(self.list_data_stream, 3, str)

    def test_txt_data_stream(self):
        self.validate_data_stream(self.txt_data_stream, 7, str)

    def test_txt_collection_data_stream(self):
        self.validate_data_stream(self.txt_collection_data_stream, 3, str)

    def test_json_collection_data_stream(self):
        self.validate_data_stream(self.json_collection_data_stream, 3, dict)

    def test_jsonl_collection_data_stream(self):
        self.validate_data_stream(self.jsonl_collection_data_stream, 7, dict)

    def test_csv_collection_data_stream(self):
        self.validate_data_stream(self.csv_collection_data_stream, 6, dict, 1)
        for data_item in self.csv_collection_data_stream:
            for key, element in data_item.items():
                self.assertIsInstance(key, str)
                self.assertIsInstance(element, str)

    def test_csv_data_stream(self):
        self.validate_data_stream(self.csv_data_stream, 3, list, 3)
        for data_item in self.csv_data_stream:
            for element in data_item:
                self.assertIsInstance(element, str)

    def test_csv_header_data_stream(self):
        self.validate_data_stream(self.csv_header_data_stream, 3, dict, 3)
        for data_item in self.csv_data_stream:
            for element in data_item:
                self.assertIsInstance(element, str)

    def test_json_array_data_stream(self):
        for data_item in self.json_data_stream:
            if isinstance(data_item, dict):
                self.assertTrue("a" in data_item)
            else:
                self.assertEqual(data_item, 4)

    def test_jsonl_array_data_stream(self):
        self.validate_data_stream(self.jsonl_data_stream, 4, dict, 2)
        for data_item in self.jsonl_data_stream:
            for element in data_item:
                self.assertIsInstance(element, str)

    def test_jsonl_control_chars_array_data_stream(self):
        self.validate_data_stream(self.jsonl_w_control_chars_data_stream, 2, dict, 2)
        for data_item in self.jsonl_data_stream:
            for element in data_item:
                self.assertIsInstance(element, str)

    def test_bad_json_stream(self):
        with self.assertRaisesRegex(ValueError, "Invalid JSON object"):
            next(iter(core_dm.DataStream.from_json_array(self.bad_json)))

    ### general data stream tests ###

    def test_iteration_is_lazy(self):
        """Verify that iteration over a simple data stream is lazy, i.e., elements are generated
        and they are requested.
        """

        def generator_func():
            generator_func.it = 0
            for i in range(10):
                yield i
                generator_func.it += 1

        counter_stream = core_dm.DataStream(generator_func)

        for count, _ in enumerate(counter_stream):
            self.assertEqual(generator_func.it, count)

    def test_generator_not_callable(self):
        """Verify that a `generator_func` that is not callable raises a `TypeError`."""
        with self.assertRaises(TypeError):
            core_dm.DataStream(42)

    def test_empty_generator_func(self):
        """Verify that an empty data stream has zero columns and zero data items."""

        def generator_func():
            return iter(range(0))

        empty_data_stream = core_dm.DataStream(generator_func)

        self.assertEqual(len(empty_data_stream), 0)

    def test_zip(self):
        """Verify that `.zip` correctly combines data stream."""
        data_stream1 = core_dm.DataStream(lambda: iter(range(3)))
        self.validate_data_stream(data_stream1, 3, int)

        data_stream2 = core_dm.DataStream(lambda: iter(range(3, 0, -1)))
        self.validate_data_stream(data_stream2, 3, int)

        zipped_streams = data_stream1.zip(data_stream2)
        self.validate_data_stream(zipped_streams, 3, tuple, 2)

        zipped_streams = core_dm.DataStream.zip(
            data_stream1, data_stream2, self.csv_data_stream
        )
        self.validate_data_stream(zipped_streams, 3, tuple, 3)

        for data_item in zipped_streams:
            self.assertIsInstance(data_item[0], int)
            self.assertIsInstance(data_item[1], int)
            self.assertIsInstance(data_item[2], list)
            self.assertEqual(len(data_item[2]), 3)

    def test_zip_different_lengths(self):
        with self.assertRaises(ValueError):
            list(core_dm.DataStream.zip(self.list_data_stream, self.txt_data_stream))

    def test_zip_is_lazy(self):
        """Verify that a simple data stream is still lazily evaluated after `.zip`."""

        def generator_func1():
            generator_func1.it = 0
            for i in range(10):
                yield i
                generator_func1.it += 1

        data_stream1 = core_dm.DataStream(generator_func1)

        def generator_func2():
            generator_func2.it = 0
            for i in range(10, 20):
                yield i
                generator_func2.it += 1

        data_stream2 = core_dm.DataStream(generator_func2)

        for count, data_item in enumerate(data_stream1.zip(data_stream2)):
            self.assertEqual(generator_func1.it, count)
            self.assertEqual(generator_func2.it, count)

    def test_zip_unequal_lengths(self):
        """Verify that attempting to zip streams with unequal length results in a value error
        when iterating over the stream.
        """
        data_stream1 = core_dm.DataStream.from_iterable(range(5))
        data_stream2 = core_dm.DataStream.from_iterable(range(10))
        zipped_stream = data_stream1.zip(data_stream2)

        with self.assertRaises(ValueError):
            list(zipped_stream)

    def test_slice(self):
        """Verify that `.__getitem__` slices data items."""

        def generator_func():
            for i in range(10):
                yield i, i + 1, i + 2, i + 3

        orig_data_stream = core_dm.DataStream(generator_func)
        self.validate_data_stream(orig_data_stream, 10, tuple, 4)

        data_stream = orig_data_stream[0]
        self.validate_data_stream(data_stream, 10, int)
        for count, data_item in enumerate(data_stream):
            self.assertEqual(data_item, count)

        data_stream = orig_data_stream[:2]
        self.validate_data_stream(data_stream, 10, tuple, 2)
        for count, data_item in enumerate(data_stream):
            self.assertEqual(data_item[0], count)
            self.assertEqual(data_item[1], count + 1)

        data_stream = orig_data_stream[1:3]
        self.validate_data_stream(data_stream, 10, tuple, 2)
        for count, data_item in enumerate(data_stream):
            self.assertEqual(data_item[0], count + 1)
            self.assertEqual(data_item[1], count + 2)

    def test_slice_is_lazy(self):
        """Verify that a simple generator is still lazily evaluated after slicing."""

        def generator_func():
            generator_func.it = 0
            for i in range(10):
                yield i, i + 1, i + 2, i + 3, i + 4
                generator_func.it += 1

        data_stream = core_dm.DataStream(generator_func)

        for count, data_item in enumerate(data_stream[0:2]):
            self.assertEqual(generator_func.it, count)

    def test_train_test_split(self):
        """Verify that a source stream can be split into two substreams with the appropriate split
        ratio and no overlapping elements
        """
        # Split the stream using a predefined seed
        src_stream = core_dm.DataStream.from_iterable(range(0, 1000))
        train_stream, test_stream = src_stream.train_test_split(
            test_split=0.25, seed=42
        )
        self.assertEqual(len(train_stream) + len(test_stream), len(src_stream))
        self.validate_data_stream(
            train_stream, 762, int
        )  # Checks for deterministic split
        self.validate_data_stream(test_stream, 238, int)  # as well as correct ratio
        # Tests for no intersection between train and test stream
        self.assertEqual(len(set(train_stream).intersection(test_stream)), 0)
        # Tests that there no duplicates in train and test streams
        self.assertEqual(len(set(train_stream)), len(train_stream))
        self.assertEqual(len(set(test_stream)), len(test_stream))

        # Verify that split is repeatable given same parameters
        train_stream2, test_stream2 = src_stream.train_test_split(
            test_split=0.25, seed=42
        )
        self.assertEqual(len(train_stream2) + len(test_stream2), len(src_stream))
        self.validate_data_stream(
            train_stream2, 762, int
        )  # Checks for deterministic split
        self.validate_data_stream(test_stream2, 238, int)  # as well as correct ratio
        self.assertTrue(all([a == b for (a, b) in zip(train_stream, train_stream2)]))
        self.assertTrue(all([a == b for (a, b) in zip(test_stream, test_stream2)]))

        # Verify that split is different given different seed
        train_stream3, test_stream3 = src_stream.train_test_split(
            test_split=0.25, seed=501
        )
        self.assertEqual(len(train_stream3) + len(test_stream3), len(src_stream))
        self.validate_data_stream(
            train_stream3, 754, int
        )  # Checks for deterministic split
        self.validate_data_stream(test_stream3, 246, int)  # as well as correct ratio

        # Verify that split is different given different test_split
        train_stream4, test_stream4 = src_stream.train_test_split(
            test_split=0.75, seed=42
        )
        self.assertEqual(len(train_stream4) + len(test_stream4), len(src_stream))
        self.validate_data_stream(
            train_stream4, 257, int
        )  # Checks for deterministic split
        self.validate_data_stream(test_stream4, 743, int)  # as well as correct ratio

    def test_shuffle(self):
        """Verify that a source stream can be shuffled"""

        # shuffle the stream using a predefined seed
        src_stream = core_dm.DataStream.from_iterable(range(0, 1000))
        shuffled_stream = src_stream.shuffle(buffer_size=200, seed=42)
        self.validate_data_stream(shuffled_stream, len(src_stream), int)
        self.assertTrue(set(shuffled_stream) == set(src_stream))

        # Verify that shuffle is repeatable given same parameters
        shuffled_stream2 = src_stream.shuffle(buffer_size=200, seed=42)
        self.validate_data_stream(shuffled_stream2, len(src_stream), int)
        self.assertTrue(list(shuffled_stream) == list(shuffled_stream2))

        # Verify that shuffle is different given different seed
        shuffled_stream3 = src_stream.shuffle(buffer_size=200, seed=452)
        self.validate_data_stream(shuffled_stream3, len(src_stream), int)
        self.assertTrue(list(shuffled_stream) != list(shuffled_stream3))

        # Verify that shuffle is different given different buffer_size
        shuffled_stream4 = src_stream.shuffle(buffer_size=500, seed=42)
        self.validate_data_stream(shuffled_stream4, len(src_stream), int)
        self.assertTrue(list(shuffled_stream) != list(shuffled_stream4))

        # Verify that shuffle is different given different buffer_size
        src_stream = core_dm.DataStream.from_iterable(range(0, 100))
        shuffled_stream = src_stream.shuffle(buffer_size=500, seed=42)
        self.validate_data_stream(shuffled_stream, len(src_stream), int)
        self.assertTrue(set(shuffled_stream) == set(src_stream))
        self.assertTrue(list(shuffled_stream) != list(src_stream))

    def test_chain(self):
        """Verify that data streams can be chained together."""
        cat_csv_pipline = core_dm.DataStream.chain(
            self.csv_data_stream, self.csv_data_stream, self.csv_data_stream
        )
        self.validate_data_stream(cat_csv_pipline, 9, list, 3)

        cat_txt_data_stream = core_dm.DataStream.chain(
            self.list_data_stream, self.txt_data_stream, self.txt_collection_data_stream
        )
        self.validate_data_stream(cat_txt_data_stream, 13, str)

    def test_chain_is_lazy(self):
        """Verify that chained data streams are lazily evaluated."""

        def generator_func1():
            for i in range(10):
                generator_func1.it = i
                yield i, i + 1

        data_stream1 = core_dm.DataStream(generator_func1)

        def generator_func2():
            for i in range(10, 20):
                generator_func1.it = i
                yield i, i + 1

        data_stream2 = core_dm.DataStream(generator_func2)

        chained_data_stream = data_stream1.chain(data_stream2)
        self.validate_data_stream(chained_data_stream, 20, tuple, 2)

        for count, data_item in enumerate(chained_data_stream):
            self.assertEqual(generator_func1.it, count)

    def test_add(self):
        """Verify that the + operator chains data streams."""
        add_data_stream = (
            self.list_data_stream
            + self.list_data_stream
            + self.txt_collection_data_stream
        )
        self.validate_data_stream(add_data_stream, 9, str)

    def test_eager_is_not_lazy(self):
        """After running `.eager`, the stream is placed into memory and no longer lazy evaluated."""

        def generator_func():
            for i in range(10):
                generator_func.it = i
                yield i, i + 1

        data_stream = core_dm.DataStream(generator_func)

        for data_item in data_stream.eager():
            self.assertEqual(generator_func.it, 9)

    def test_map(self):
        """Verify that a simple application of the map functino works as expected."""
        # stream over integer range
        int_stream = core_dm.DataStream(lambda: iter(range(10)))

        # map function returns a tuple with both the original value and also the value plus one
        int_plus_one_stream = int_stream.map(
            lambda data_item: (data_item, data_item + 1)
        )

        for count, data_item in enumerate(int_plus_one_stream):
            self.assertEqual(data_item[0], count)
            self.assertEqual(data_item[1], count + 1)

    def test_map_is_lazy(self):
        """Verify that lazy evaluation occurs after a `.map`."""

        def generator_func():
            generator_func.it = 0
            for i in range(10):
                yield i
                generator_func.it += 1

        int_stream = core_dm.DataStream(generator_func)

        for count, data_item in enumerate(int_stream):
            self.assertEqual(data_item, count)
            self.assertEqual(generator_func.it, count)

    def test_filter_empty_lines(self):
        """Verify that a simple filter can remove empty lines of text."""
        no_empty_txt_stream = self.txt_data_stream.filter()
        self.assertEqual(len(no_empty_txt_stream), 4)

    def test_filter_is_lazy(self):
        """Verify that lazy evaluation occurs after a `.filter`."""

        def generator_func():
            generator_func.it = 0
            for i in range(10):
                yield i
                generator_func.it += 1

        int_stream = core_dm.DataStream(generator_func)
        odd_int_stream = int_stream.filter(lambda data_item: data_item % 2)
        self.validate_data_stream(odd_int_stream, 5, int)

        for count, data_item in enumerate(odd_int_stream):
            self.assertEqual(data_item, count * 2 + 1)

    def test_dummy_stream(self):
        """Verify that we can use streams to generate dummy predictions."""
        text_stream = self.csv_data_stream[0]
        self.validate_data_stream(text_stream, 3, str)

        dummy_model = caikit.core.load(os.path.join(self.fixtures_dir, "dummy_module"))
        # Map to data stream of SampleInputType for model
        sample_type_data_stream = text_stream.map(
            lambda text: SampleInputType(name=text)
        )
        dummy_stream = dummy_model.stream(sample_type_data_stream)
        self.validate_data_stream(dummy_stream, 3, SampleOutputType)

    def test_only_one_stream_allowed(self):  # , mock_run):
        """Verify that we can't call module.stream() with multiple DataStream args."""
        text_stream = self.csv_data_stream[0]
        self.validate_data_stream(text_stream, 3, str)

        dummy_model = caikit.core.load(os.path.join(self.fixtures_dir, "dummy_module"))
        with self.assertRaises(ValueError):
            # NOTE: Even though this is too many args for .run() to handle, it's okay; this
            # will fail before it gets to .run(). In the event that .run() is called with the
            # wrong number of positional arguments, it'll throw a TypeError, which fails this test.
            dummy_model.stream(text_stream, text_stream)

    def test_pipe_stream(self):
        """Verify that we can use streams to generate dummy predictions using `|` syntax."""
        dummy_model = caikit.core.load(os.path.join(self.fixtures_dir, "dummy_module"))

        dummy_stream = self.sample_type_data_stream | dummy_model
        self.validate_data_stream(dummy_stream, 7, SampleOutputType)

    ### generic file loader tests

    def test_from_file_can_handle_a_json_file(self):
        json_stream = core_dm.DataStream.from_file(
            os.path.join(self.samples_path, "sample.json")
        )
        for data_item in self.json_data_stream:
            if isinstance(data_item, dict):
                self.assertTrue("a" in data_item)
            else:
                self.assertEqual(data_item, 4)

    ### Augmentor tests

    def test_augment_enforces_augmentor_type_checking(self):
        """Ensure that we need to pass an augmentor to augment a stream."""
        input_stream = core_dm.DataStream.from_iterable([1, 0, 3, 2])
        self.assertRaises(TypeError, input_stream.augment, "garbage", 1)

    def test_augment_validates_aug_cycles_properly(self):
        """Ensure that we properly validate aug_cycles when augmenting."""
        input_stream = core_dm.DataStream.from_iterable([1, 0, 3, 2])
        aug = build_test_augmentor(True)
        self.assertRaises(TypeError, input_stream.augment, aug, "Garbage")
        self.assertRaises(ValueError, input_stream.augment, aug, 0)

    def test_augment_valides_enforce_determinism_properly(self):
        """Make sure that we throw if enforce_determinism is not a bool."""
        input_stream = core_dm.DataStream.from_iterable([1, 0, 3, 2])
        aug = build_test_augmentor(True)
        with self.assertRaises(TypeError):
            input_stream.augment(aug, 2, enforce_determinism=4)

    def test_augment_validates_post_func_callback(self):
        """Ensure that the post augmentation callback needs to be a function."""
        input_stream = core_dm.DataStream.from_iterable([1, 0, 3, 2])
        aug = build_test_augmentor(True)
        with self.assertRaises(TypeError):
            input_stream.augment(aug, 2, post_augment_func="not a function")

    def test_augmentor_that_returns_none_when_expected(self):
        """Test that if an augmentor is allowed to return None, none values drop."""
        input_sequence = [1, 0, 3, 2]
        output_sequence = [1, 0, 3, 2, 6, 8, 7]
        aug = build_test_augmentor(True)
        actual_sequence = list(
            core_dm.DataStream.from_iterable(input_sequence).augment(aug, 1)
        )
        self.assertEqual(output_sequence, actual_sequence)

    def test_augmentor_that_returns_none_when_not_expected(self):
        """Test that we raise if an augmentor produces None when it is not supposed to."""
        input_sequence = [13]
        aug = build_test_augmentor(False)
        # Explode because we said this augmentor should never raise None, and we pass unlucky 13 :(
        with self.assertRaises(TypeError):
            list(core_dm.DataStream.from_iterable(input_sequence).augment(aug, 1))

    def test_index_targeting(self):
        """Test that index targeting works as expected."""
        input_sequence = [(1, "foo"), (0, "bar")]
        output_sequence = [(1, "foo"), (0, "bar"), (6, "foo"), (5, "bar")]
        input_stream = core_dm.DataStream.from_iterable(input_sequence)
        aug = build_test_augmentor(False)
        actual_sequence = list(input_stream.augment(aug, 1, augment_index=0))
        self.assertEqual(output_sequence, actual_sequence)

    def test_out_of_bounds_index_target(self):
        """Test that if we specify an out of bounds index target, we explode correctly."""
        input_sequence = [(1, "foo"), (0, "bar")]
        input_stream = core_dm.DataStream.from_iterable(input_sequence)
        aug = build_test_augmentor(False)
        with self.assertRaises(IndexError):
            list(input_stream.augment(aug, 1, augment_index=2))

    def test_post_augment_func(self):
        """Test that we can apply a post augmentation callback."""
        callback = lambda val: str(val)
        input_sequence = [1, 0, 3, 2]
        output_sequence = ["1", "0", "3", "2", "6", "5", "8", "7"]
        input_stream = core_dm.DataStream.from_iterable(input_sequence)
        aug = build_test_augmentor(False)
        actual_sequence = list(input_stream.augment(aug, 1, post_augment_func=callback))
        self.assertEqual(output_sequence, actual_sequence)

    def test_post_augment_func_with_index_targeting(self):
        """Test that we can apply a post augmentation callback with targeted indices."""
        input_sequence = [(1, "foo"), (0, "bar")]
        # Callback will stringify the values that we get back
        callback = lambda val: str(val)
        output_sequence = [("1", "foo"), ("0", "bar"), ("6", "foo"), ("5", "bar")]
        input_stream = core_dm.DataStream.from_iterable(input_sequence)
        aug = build_test_augmentor(False)
        actual_sequence = list(
            input_stream.augment(aug, 1, augment_index=0, post_augment_func=callback)
        )
        self.assertEqual(output_sequence, actual_sequence)
