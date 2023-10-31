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

# Third Party
import pytest

# Local
from caikit.core.augmentors import AugmentorBase
from caikit.core.data_model import DataStream
from sample_lib.data_model.sample import SampleInputType, SampleOutputType
import caikit.core

###############################
# Helper functions and fixtures
###############################


def validate_data_stream(data_stream, length, data_item_type, data_item_length=None):
    # Verify lengths and types
    assert isinstance(data_stream, DataStream)
    assert len(data_stream) == length
    assert sum(1 for _ in data_stream) == length

    for data_item in data_stream:
        assert isinstance(data_item, data_item_type)
        if data_item_length is not None:
            assert len(data_item) == data_item_length


def build_test_augmentor(produces_none):
    class TestStreamAugmentor(AugmentorBase):
        """A 'normal' augmentor, which when leveraged correctly, always produces an output per input."""

        augmentor_type = int
        # Indicates that this class should not be picked up by augmentor discovery utilities.
        is_test_augmentor = True

        def __init__(self):
            super().__init__(random_seed=1001, produces_none=produces_none)

        def _augment(self, obj):
            """If produces_none is True, filters out falsy values, i.e., 0s, otherwise adds
            five to the original input.

            Also return None if we pass 13 and set produces_none=False, as a sad test case.
            This should raise a TypeError within the augmentor base class.
            """
            if produces_none and not obj:
                return None
            if not produces_none and obj == 13:
                return None
            return obj + 5

    return TestStreamAugmentor()


## Local fixtures ###
@pytest.fixture
def list_data_stream():
    yield DataStream.from_iterable(["hello", "world", "!"])


###############################
# Tests on various data formats
###############################


def test_list_data_stream(list_data_stream):
    validate_data_stream(list_data_stream, 3, str)


def test_txt_data_stream(sample_text_file):
    validate_data_stream(DataStream.from_txt(sample_text_file), 7, str)


def test_txt_collection_data_stream(sample_text_collection):
    validate_data_stream(DataStream.from_txt_collection(sample_text_collection), 3, str)


def test_json_collection_data_stream(sample_json_collection):
    validate_data_stream(
        DataStream.from_json_collection(sample_json_collection), 3, dict
    )


def test_jsonl_collection_data_stream(sample_jsonl_collection):
    validate_data_stream(
        DataStream.from_jsonl_collection(sample_jsonl_collection), 7, dict
    )


def test_csv_collection_data_stream(sample_csv_collection):
    csv_collection_data_stream = DataStream.from_csv_collection(sample_csv_collection)
    validate_data_stream(csv_collection_data_stream, 6, dict, 2)
    for data_item in csv_collection_data_stream:
        for key, element in data_item.items():
            assert isinstance(key, str)
            assert isinstance(element, str)


def test_csv_data_stream(sample_csv_file_no_headers):
    csv_data_stream = DataStream.from_csv(sample_csv_file_no_headers)
    validate_data_stream(csv_data_stream, 3, list, 2)
    for data_item in csv_data_stream:
        for element in data_item:
            assert isinstance(element, str)


def test_csv_header_data_stream(sample_csv_file):
    csv_header_data_stream = DataStream.from_header_csv(sample_csv_file)
    validate_data_stream(csv_header_data_stream, 3, dict, 2)
    for data_item in csv_header_data_stream:
        for element in data_item:
            assert isinstance(element, str)


def test_json_array_data_stream(sample_json_file):
    json_data_stream = DataStream.from_json_array(sample_json_file)
    with open(sample_json_file) as f:
        assert list(json_data_stream) == json.load(f)


def test_jsonl_array_data_stream(sample_jsonl_file):
    jsonl_data_stream = DataStream.from_jsonl(sample_jsonl_file)
    validate_data_stream(jsonl_data_stream, 4, dict, 2)
    for data_item in jsonl_data_stream:
        for element in data_item:
            assert isinstance(element, str)


def test_jsonl_control_chars_array_data_stream(jsonl_with_control_chars):
    data_stream = DataStream.from_jsonl(jsonl_with_control_chars)
    validate_data_stream(data_stream, 2, dict, 2)
    for data_item in data_stream:
        for element in data_item:
            assert isinstance(element, str)


def test_data_stream_from_jsonl_is_pickleable(tmp_path):
    tmpdir = str(tmp_path)

    data = [1, 2, 3, 4, 5, 6]
    filepath = os.path.join(tmpdir, "foo.jsonl")
    with open(filepath, "w") as f:
        json.dump(data, f)

    stream = DataStream.from_jsonl(filepath)

    pre_pickle_vals = list(stream)
    pickled_stream = pickle.loads(pickle.dumps(stream))
    post_pickle_vals = list(pickled_stream)

    assert pre_pickle_vals == post_pickle_vals
    # Interesting: Technically this is a stream of length 1 where the one element is [1,2,3,4,5,6]
    validate_data_stream(pickled_stream, 1, list)


def test_data_stream_from_json_is_pickleable(tmp_path):
    tmpdir = str(tmp_path)

    data = [1, 2, 3, 4, 5, 6]
    filepath = os.path.join(tmpdir, "foo.json")
    with open(filepath, "w") as f:
        json.dump(data, f)

    stream = DataStream.from_json_array(filepath)

    pre_pickle_vals = list(stream)
    pickled_stream = pickle.loads(pickle.dumps(stream))
    post_pickle_vals = list(pickled_stream)

    assert pre_pickle_vals == post_pickle_vals
    validate_data_stream(pickled_stream, 6, int)


def test_bad_json_stream(tmp_path):
    file_path = os.path.join(str(tmp_path), "bad.json")
    with open(file_path, "w") as fp:
        fp.write(
            """
            text
            Some
            { "a" : 1}
            "a": 1
        """
        )
    with pytest.raises(ValueError, match="Invalid JSON object"):
        DataStream.from_json_array(file_path).peek()


def test_non_json_array_stream(tmp_path):
    file_path = os.path.join(str(tmp_path), "non_array.json")
    with open(file_path, "w") as fp:
        fp.write(
            """
        {
            "number": 1,
            "label": "foo"
        }
        """
        )
    with pytest.raises(ValueError, match="Non-array JSON object"):
        DataStream.from_json_array(file_path).peek()


def test_bad_multipart_file(tmp_path):
    multipart_file = os.path.join(str(tmp_path), "bad_multipart")
    with open(multipart_file, "w") as fp:
        fp.write("content")
        fp.write("\n")
        fp.write("--")
        fp.write("arbitrary")
    with pytest.raises(ValueError, match="file is not multipart"):
        DataStream.from_multipart_file(multipart_file)


def test_invalid_multipart_boundary(tmp_path):
    file_path = os.path.join(str(tmp_path), "bad.multipart")
    with open(file_path, "w") as fp:
        fp.write(
            """
Content-Type: multipart/form-data; boundary=foo

--foo
Content-Disposition: form-data; name=""; filename="my_training_data.json"
Content-Type: application/json

[
    {
        "number": 1,
        "label": "small"
    },
    {
        "number": 1000,
        "label": "large"
    }
]

--bar--
            """
        )
    with pytest.raises(ValueError, match="Invalid form-data"):
        DataStream.from_multipart_file(file_path).peek()


def test_from_file_can_handle_a_json_file(sample_json_file):
    json_stream = DataStream.from_file(sample_json_file)
    validate_data_stream(json_stream, 2, dict)


def test_from_multipart_file_json(sample_multipart_json):
    multipart_stream = DataStream.from_multipart_file(sample_multipart_json)
    validate_data_stream(multipart_stream, 2, dict)


def test_from_multipart_file_csv(sample_multipart_csv):
    multipart_stream = DataStream.from_multipart_file(sample_multipart_csv)
    validate_data_stream(multipart_stream, 2, dict)


def test_from_multipart_file_unsupported_content_type(tmp_path):
    multipart_file = os.path.join(str(tmp_path), "multipart")

    # Still need to create valid multipart file
    with open(multipart_file, "w") as fp:
        fp.write("--")
        fp.write("arbitrary")
        fp.writelines(
            [
                "\n",
                f'Content-Disposition: form-data; name=""; filename="file"\n',
                "Content-Type: text/plain\n",
                "\n",
            ]
        )
        fp.write("content")
        fp.write("\n")
        fp.write("--")
        fp.write("arbitrary")
        fp.write("--")
    with pytest.raises(ValueError, match="Unsupported content type: text/plain"):
        DataStream.from_multipart_file(multipart_file)


###########################
# General data stream tests
###########################


def test_iteration_is_lazy():
    """Verify that iteration over a simple data stream is lazy, i.e., elements are generated
    as they are requested.
    """

    def generator_func():
        generator_func.it = 0
        for i in range(10):
            yield i
            generator_func.it += 1

    counter_stream = DataStream(generator_func)

    for count, _ in enumerate(counter_stream):
        assert generator_func.it == count


def test_generator_not_callable():
    """Verify that a `generator_func` that is not callable raises a `TypeError`."""
    with pytest.raises(TypeError):
        DataStream(42)


def test_empty_generator_func():
    """Verify that an empty data stream has zero columns and zero data items."""

    def generator_func():
        return iter(range(0))

    empty_data_stream = DataStream(generator_func)

    assert len(empty_data_stream) == 0


def test_zip(sample_csv_file_no_headers):
    """Verify that `.zip` correctly combines data stream."""
    data_stream1 = DataStream(lambda: iter(range(3)))
    validate_data_stream(data_stream1, 3, int)

    data_stream2 = DataStream(lambda: iter(range(3, 0, -1)))
    validate_data_stream(data_stream2, 3, int)

    zipped_streams = data_stream1.zip(data_stream2)
    validate_data_stream(zipped_streams, 3, tuple, 2)

    zipped_streams = DataStream.zip(
        data_stream1, data_stream2, DataStream.from_csv(sample_csv_file_no_headers)
    )
    validate_data_stream(zipped_streams, 3, tuple, 3)

    for data_item in zipped_streams:
        assert isinstance(data_item[0], int)
        assert isinstance(data_item[1], int)
        assert isinstance(data_item[2], list)
        assert len(data_item[2]) == 2


def test_zip_different_lengths(sample_text_file, list_data_stream):
    with pytest.raises(ValueError):
        list(DataStream.zip(list_data_stream, DataStream.from_txt(sample_text_file)))


def test_zip_is_lazy():
    """Verify that a simple data stream is still lazily evaluated after `.zip`."""

    def generator_func1():
        generator_func1.it = 0
        for i in range(10):
            yield i
            generator_func1.it += 1

    data_stream1 = DataStream(generator_func1)

    def generator_func2():
        generator_func2.it = 0
        for i in range(10, 20):
            yield i
            generator_func2.it += 1

    data_stream2 = DataStream(generator_func2)

    for count, _ in enumerate(data_stream1.zip(data_stream2)):
        assert generator_func1.it == count
        assert generator_func2.it == count


def test_zip_unequal_lengths():
    """Verify that attempting to zip streams with unequal length results in a value error
    when iterating over the stream.
    """
    data_stream1 = DataStream.from_iterable(range(5))
    data_stream2 = DataStream.from_iterable(range(10))
    zipped_stream = data_stream1.zip(data_stream2)

    with pytest.raises(ValueError):
        list(zipped_stream)


def test_slice():
    """Verify that `.__getitem__` slices data items."""

    def generator_func():
        for i in range(10):
            yield i, i + 1, i + 2, i + 3

    orig_data_stream = DataStream(generator_func)
    validate_data_stream(orig_data_stream, 10, tuple, 4)

    data_stream = orig_data_stream[0]
    validate_data_stream(data_stream, 10, int)
    for count, data_item in enumerate(data_stream):
        assert data_item == count

    data_stream = orig_data_stream[:2]
    validate_data_stream(data_stream, 10, tuple, 2)
    for count, data_item in enumerate(data_stream):
        assert data_item[0] == count
        assert data_item[1] == count + 1

    data_stream = orig_data_stream[1:3]
    validate_data_stream(data_stream, 10, tuple, 2)
    for count, data_item in enumerate(data_stream):
        assert data_item[0] == count + 1
        assert data_item[1] == count + 2


def test_slice_is_lazy():
    """Verify that a simple generator is still lazily evaluated after slicing."""

    def generator_func():
        generator_func.it = 0
        for i in range(10):
            yield i, i + 1, i + 2, i + 3, i + 4
            generator_func.it += 1

    data_stream = DataStream(generator_func)

    for count, _ in enumerate(data_stream[0:2]):
        assert generator_func.it == count


def test_train_test_split():
    """Verify that a source stream can be split into two substreams with the appropriate split
    ratio and no overlapping elements
    """
    # Split the stream using a predefined seed
    src_stream = DataStream.from_iterable(range(0, 1000))
    train_stream, test_stream = src_stream.train_test_split(test_split=0.25, seed=42)
    assert len(train_stream) + len(test_stream) == len(src_stream)
    validate_data_stream(train_stream, 762, int)  # Checks for deterministic split
    validate_data_stream(test_stream, 238, int)  # as well as correct ratio
    # Tests for no intersection between train and test stream
    assert len(set(train_stream).intersection(test_stream)) == 0
    # Tests that there no duplicates in train and test streams
    assert len(set(train_stream)) == len(train_stream)
    assert len(set(test_stream)) == len(test_stream)

    # Verify that split is repeatable given same parameters
    train_stream2, test_stream2 = src_stream.train_test_split(test_split=0.25, seed=42)
    assert len(train_stream2) + len(test_stream2) == len(src_stream)
    validate_data_stream(train_stream2, 762, int)  # Checks for deterministic split
    validate_data_stream(test_stream2, 238, int)  # as well as correct ratio
    assert all([a == b for (a, b) in zip(train_stream, train_stream2)])
    assert all([a == b for (a, b) in zip(test_stream, test_stream2)])

    # Verify that split is different given different seed
    train_stream3, test_stream3 = src_stream.train_test_split(test_split=0.25, seed=501)
    assert len(train_stream3) + len(test_stream3) == len(src_stream)
    validate_data_stream(train_stream3, 754, int)  # Checks for deterministic split
    validate_data_stream(test_stream3, 246, int)  # as well as correct ratio

    # Verify that split is different given different test_split
    train_stream4, test_stream4 = src_stream.train_test_split(test_split=0.75, seed=42)
    assert len(train_stream4) + len(test_stream4) == len(src_stream)
    validate_data_stream(train_stream4, 257, int)  # Checks for deterministic split
    validate_data_stream(test_stream4, 743, int)  # as well as correct ratio


def test_shuffle():
    """Verify that a source stream can be shuffled"""

    # Shuffle the stream using a predefined seed
    src_stream = DataStream.from_iterable(range(0, 1000))
    shuffled_stream = src_stream.shuffle(buffer_size=200, seed=42)
    validate_data_stream(shuffled_stream, len(src_stream), int)
    assert set(shuffled_stream) == set(src_stream)

    # Verify that shuffle is repeatable given same parameters
    shuffled_stream2 = src_stream.shuffle(buffer_size=200, seed=42)
    validate_data_stream(shuffled_stream2, len(src_stream), int)
    assert list(shuffled_stream) == list(shuffled_stream2)

    # Verify that shuffle is different given different seed
    shuffled_stream3 = src_stream.shuffle(buffer_size=200, seed=452)
    validate_data_stream(shuffled_stream3, len(src_stream), int)
    assert list(shuffled_stream) != list(shuffled_stream3)

    # Verify that shuffle is different given different buffer_size
    shuffled_stream4 = src_stream.shuffle(buffer_size=500, seed=42)
    validate_data_stream(shuffled_stream4, len(src_stream), int)
    assert list(shuffled_stream) != list(shuffled_stream4)

    # Verify that shuffle is different given different buffer_size
    src_stream = DataStream.from_iterable(range(0, 100))
    shuffled_stream = src_stream.shuffle(buffer_size=500, seed=42)
    validate_data_stream(shuffled_stream, len(src_stream), int)
    assert set(shuffled_stream) == set(src_stream)
    assert list(shuffled_stream) != list(src_stream)


def test_chain(
    sample_csv_file_no_headers,
    sample_text_file,
    sample_text_collection,
    list_data_stream,
):
    csv_data_stream = DataStream.from_csv(sample_csv_file_no_headers)
    """Verify that data streams can be chained together."""
    cat_csv_pipline = DataStream.chain(
        csv_data_stream, csv_data_stream, csv_data_stream
    )
    validate_data_stream(cat_csv_pipline, 9, list, 2)

    cat_txt_data_stream = DataStream.chain(
        list_data_stream,
        DataStream.from_txt(sample_text_file),
        DataStream.from_txt_collection(sample_text_collection),
    )
    validate_data_stream(cat_txt_data_stream, 13, str)


def test_chain_is_lazy():
    """Verify that chained data streams are lazily evaluated."""

    def generator_func1():
        for i in range(10):
            generator_func1.it = i
            yield i, i + 1

    data_stream1 = DataStream(generator_func1)

    def generator_func2():
        for i in range(10, 20):
            generator_func1.it = i
            yield i, i + 1

    data_stream2 = DataStream(generator_func2)

    chained_data_stream = data_stream1.chain(data_stream2)
    validate_data_stream(chained_data_stream, 20, tuple, 2)

    for count, _ in enumerate(chained_data_stream):
        assert generator_func1.it == count


def test_add(sample_text_collection, list_data_stream):
    """Verify that the + operator chains data streams."""
    add_data_stream = (
        list_data_stream
        + list_data_stream
        + DataStream.from_txt_collection(sample_text_collection)
    )
    validate_data_stream(add_data_stream, 9, str)


def test_eager_is_not_lazy():
    """After running `.eager`, the stream is placed into memory and no longer lazy evaluated."""

    def generator_func():
        for i in range(10):
            generator_func.it = i
            yield i, i + 1

    data_stream = DataStream(generator_func)

    for _ in data_stream.eager():
        assert generator_func.it == 9


def test_map():
    """Verify that a simple application of the map functino works as expected."""
    # stream over integer range
    int_stream = DataStream(lambda: iter(range(10)))

    # map function returns a tuple with both the original value and also the value plus one
    int_plus_one_stream = int_stream.map(lambda data_item: (data_item, data_item + 1))

    for count, data_item in enumerate(int_plus_one_stream):
        assert data_item[0] == count
        assert data_item[1] == count + 1


def test_map_is_lazy():
    """Verify that lazy evaluation occurs after a `.map`."""

    def generator_func():
        generator_func.it = 0
        for i in range(10):
            yield i
            generator_func.it += 1

    int_stream = DataStream(generator_func)

    for count, data_item in enumerate(int_stream):
        assert data_item == count
        assert generator_func.it == count


def test_filter_empty_lines(sample_text_file):
    """Verify that a simple filter can remove empty lines of text."""
    no_empty_txt_stream = DataStream.from_txt(sample_text_file).filter()
    assert len(no_empty_txt_stream) == 4


def test_filter_is_lazy():
    """Verify that lazy evaluation occurs after a `.filter`."""

    def generator_func():
        generator_func.it = 0
        for i in range(10):
            yield i
            generator_func.it += 1

    int_stream = DataStream(generator_func)
    odd_int_stream = int_stream.filter(lambda data_item: data_item % 2)
    validate_data_stream(odd_int_stream, 5, int)

    for count, data_item in enumerate(odd_int_stream):
        assert data_item == count * 2 + 1


def test_dummy_stream(good_model_path, sample_csv_file):
    """Verify that we can use streams to generate dummy predictions."""
    dummy_model = caikit.core.load(good_model_path)

    stream_len = 7
    sample_input_stream = DataStream.from_iterable(
        [SampleInputType(name="gabe")] * stream_len
    )
    dummy_stream = dummy_model.stream(sample_input_stream)
    validate_data_stream(dummy_stream, stream_len, SampleOutputType)


def test_only_one_stream_allowed(
    sample_csv_file_no_headers, good_model_path
):  # , mock_run):
    """Verify that we can't call module.stream() with multiple DataStream args."""
    stream = DataStream.from_iterable([1, 2, 3, 4, 5])

    dummy_model = caikit.core.load(good_model_path)
    with pytest.raises(ValueError):
        # NOTE: Even though this is too many args for .run() to handle, it's okay; this
        # will fail before it gets to .run(). In the event that .run() is called with the
        # wrong number of positional arguments, it'll throw a TypeError, which fails this test.
        dummy_model.stream(stream, stream)


def test_pipe_stream(good_model_path):
    """Verify that we can use streams to generate dummy predictions using `|` syntax."""
    dummy_model = caikit.core.load(good_model_path)
    stream_len = 7
    sample_input_stream = DataStream.from_iterable(
        [SampleInputType(name="gabe")] * stream_len
    )

    dummy_stream = sample_input_stream | dummy_model
    validate_data_stream(dummy_stream, stream_len, SampleOutputType)


##################
# Augmentor tests
##################


def test_augment_enforces_augmentor_type_checking():
    """Ensure that we need to pass an augmentor to augment a stream."""
    input_stream = DataStream.from_iterable([1, 0, 3, 2])
    with pytest.raises(TypeError):
        input_stream.augment("garbage", 1)


def test_augment_validates_aug_cycles_properly():
    """Ensure that we properly validate aug_cycles when augmenting."""
    input_stream = DataStream.from_iterable([1, 0, 3, 2])
    aug = build_test_augmentor(True)
    with pytest.raises(TypeError):
        input_stream.augment(aug, "Garbage")
    with pytest.raises(ValueError):
        input_stream.augment(aug, 0)


def test_augment_valides_enforce_determinism_properly():
    """Make sure that we throw if enforce_determinism is not a bool."""
    input_stream = DataStream.from_iterable([1, 0, 3, 2])
    aug = build_test_augmentor(True)
    with pytest.raises(TypeError):
        input_stream.augment(aug, 2, enforce_determinism=4)


def test_augment_validates_post_func_callback():
    """Ensure that the post augmentation callback needs to be a function."""
    input_stream = DataStream.from_iterable([1, 0, 3, 2])
    aug = build_test_augmentor(True)
    with pytest.raises(TypeError):
        input_stream.augment(aug, 2, post_augment_func="not a function")


def test_augmentor_that_returns_none_when_expected():
    """Test that if an augmentor is allowed to return None, none values drop."""
    input_sequence = [1, 0, 3, 2]
    output_sequence = [1, 0, 3, 2, 6, 8, 7]
    aug = build_test_augmentor(True)
    actual_sequence = list(DataStream.from_iterable(input_sequence).augment(aug, 1))
    assert output_sequence == actual_sequence


def test_augmentor_that_returns_none_when_not_expected():
    """Test that we raise if an augmentor produces None when it is not supposed to."""
    input_sequence = [13]
    aug = build_test_augmentor(False)
    # Explode because we said this augmentor should never raise None, and we pass unlucky 13 :(
    with pytest.raises(TypeError):
        list(DataStream.from_iterable(input_sequence).augment(aug, 1))


def test_index_targeting():
    """Test that index targeting works as expected."""
    input_sequence = [(1, "foo"), (0, "bar")]
    output_sequence = [(1, "foo"), (0, "bar"), (6, "foo"), (5, "bar")]
    input_stream = DataStream.from_iterable(input_sequence)
    aug = build_test_augmentor(False)
    actual_sequence = list(input_stream.augment(aug, 1, augment_index=0))
    assert output_sequence == actual_sequence


def test_out_of_bounds_index_target():
    """Test that if we specify an out of bounds index target, we explode correctly."""
    input_sequence = [(1, "foo"), (0, "bar")]
    input_stream = DataStream.from_iterable(input_sequence)
    aug = build_test_augmentor(False)
    with pytest.raises(IndexError):
        list(input_stream.augment(aug, 1, augment_index=2))


def test_post_augment_func():
    """Test that we can apply a post augmentation callback."""
    callback = lambda val: str(val)
    input_sequence = [1, 0, 3, 2]
    output_sequence = ["1", "0", "3", "2", "6", "5", "8", "7"]
    input_stream = DataStream.from_iterable(input_sequence)
    aug = build_test_augmentor(False)
    actual_sequence = list(input_stream.augment(aug, 1, post_augment_func=callback))
    assert output_sequence == actual_sequence


def test_post_augment_func_with_index_targeting():
    """Test that we can apply a post augmentation callback with targeted indices."""
    input_sequence = [(1, "foo"), (0, "bar")]
    # Callback will stringify the values that we get back
    callback = lambda val: str(val)
    output_sequence = [("1", "foo"), ("0", "bar"), ("6", "foo"), ("5", "bar")]
    input_stream = DataStream.from_iterable(input_sequence)
    aug = build_test_augmentor(False)
    actual_sequence = list(
        input_stream.augment(aug, 1, augment_index=0, post_augment_func=callback)
    )
    assert output_sequence == actual_sequence
