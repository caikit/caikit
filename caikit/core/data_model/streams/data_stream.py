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


"""Data streams for lazily loading, munging and passing data through multiple modules.
"""

# Standard
from collections.abc import Iterable
from glob import glob
from typing import Dict, Generic, List, Tuple, TypeVar, Union
import collections
import csv
import itertools
import json
import os
import random
import typing

# Third Party
import ijson

# First Party
import alog

# Local
from ...augmentors import AugmentorBase
from ...toolkit import fileio
from ...toolkit.errors import error_handler

log = alog.use_channel("DATSTRM")
error = error_handler.get(log)

T = TypeVar("T")


# ghart: These public methods are all needed. This class is essentially its own factory, so these
# are all the different ways of coercing different data sources into a common stream class
# pylint: disable=too-many-public-methods
class DataStream(Generic[T]):
    """A data stream is a iterable container class that is reentrant in the sense that it can be
    iterated over multiple times.  The items produced by a data stream may be any python object
    and are called data items.  The data items produced by an iterator over a data stream are
    generated lazily (unless the `.eager` method is called) so that each data item in a series of
    data streams is produced as it is accessed.  This allows processing datasets that are too large
    to fit into memory.  A number of functional style methods are provided for manipulating and
    munging data streams and the `.stream` method on modules can also be used to
    process data streams.

    The `DataStream` class is really just a generic wrapper around functions that produce python
    iterators or generators.
    """

    def __init__(self, generator_func, *args, **kwargs):
        """Create a new `DataStream` from a function that creates a python generator or iterator
        over the desired data items.

        Args:
            generator_func (callable(*args, **kwargs)): A function that, when
                called, either (a) constructs a generator or (b) returns a
                python iterator that yields data items, which may be any python
                or data model object.  Each time `generator_func` is called, it
                must recreate the same generator/iterator.  `generator_func`
                must also produce its elements lazily. If `generator_func`
                returns, say a list or tuple, then all of the data will be
                loaded into memory immediately instead of lazily.
                `generator_func` is invoked every time that a `DataStream` is
                iterated over, i.e., when `__iter__` is called.

            args, kwargs: Additional arguments passed to `generator_func`.  These are passed every
                time that `generator_func` is called, i.e., every time we iterate over the data
                stream. These arguments are generally useful for passing arguments to an initial
                data loader function (see `.from_csv` for an example).  In order to retain other
                variables in a `generator_func` consider relying on closures instead of arguments.

        Notes:
            The constructor of `DataStream` is not usually invoked directly.  The typical use case
            is to construct a data stream using one of the `.from_` class methods or else from the
            `.stream` method of a module or by extending the `DataStream` class.

            Lexical closures, generators and iterators are all important to understand when writing
            a new `generator_func`.  Consider reviewing these topics before writing custom generator
            functions.
        """
        if not callable(generator_func):
            error(
                "<COR83886275E>",
                TypeError("Data stream `generator_func` is not callable."),
            )

        self.generator_func = generator_func
        self.generator_args, self.generator_kwargs = args, kwargs
        self._length = None

    @classmethod
    def from_iterable(cls, data: typing.Iterable[T]) -> "DataStream[T]":
        """Create a new data stream from a python iterable, such as a list or tuple.  This data
        stream produces a single data item for each element of the iterable..

        Args:
            data (iterable): A list or tuple or other python iterable used to
                construct a new data stream where each data item contains a
                single data item.

        Returns:
            DataStream: A new data stream that produces data items from the
                elements of `data`.

        Examples:
            >>> list_stream = DataStream.from_iterable([1, 2, 3])
            >>> for data_item in list_stream:
            >>>     print(data_item)
            1
            2
            3
        """
        error.type_check("<COR88684982E>", Iterable, data=data)
        return cls(cls._from_iterable_generator, data)

    @classmethod
    def _from_iterable_generator(cls, data: typing.Iterable[T]) -> typing.Iterator[T]:
        return iter(data)

    @classmethod
    def from_jsonl(cls, filename: str) -> "DataStream[Dict]":
        """Creates a new data stream from a path to a file with JSON lines array, where
        each line is a valid JSON (python dict)

        Args:
            filename (str): A path to a utf8 encode text file with JSON lines
                array, where each line is a valid JSON (python dict)

        Returns:
            DataStream: A new data stream that produces python dict items each
                containing a single JSON object corresponding to each line

        Notes:
            This class method returns a data stream over the valid JSON objects and each
            JSON object is on one line.

            https://jsonlines.org/

        Examples:
            For a JSON lines file that looks like:
                {"name": "Gilbert", "wins": [["straight", "7♣"], ["one pair", "10♥"]]}
                {"name": "Alexa", "wins": [["two pair", "4♠"], ["two pair", "9♠"]]}
                {"name": "May", "wins": []}
                {"name": "Deloise", "wins": [["three of a kind", "5♣"]]}

            >>> jsonl_data_stream = DataStream.from_jsonl('sample.jsonl')
            >>> for data_item in jsonl_data_stream:
            >>>     print(data_item)
            {'name': 'Gilbert', 'wins': [['straight', '7♣'], ['one pair', '10♥']]}
            {'name': 'Alexa', 'wins': [['two pair', '4♠'], ['two pair', '9♠']]}
            {'name': 'May', 'wins': []}
            {'name': 'Deloise', 'wins': [['three of a kind', '5♣']]}

        """
        error.file_check("<COR32600575E>", filename)

        return cls(cls._from_jsonl_generator, filename)

    @classmethod
    def _from_jsonl_generator(cls, filename):
        with open(filename, mode="rb") as json_fh:
            log.debug2("Loading JSON array file:  %s", filename)
            lines = json_fh.readlines()

            try:
                for line in lines:
                    if line.strip():  # ignore empty lines
                        yield json.loads(line)
            except json.JSONDecodeError as e:
                error(
                    "<COR55596551E>",
                    ValueError(f"Invalid JSON object in `{line}`, error: {e.msg}"),
                )
            except TypeError:
                error(
                    "<COR35596551E>",
                    ValueError("Invalid JSON object in `{}`".format(line)),
                )

    @classmethod
    def from_json_array(cls, filename: str) -> "DataStream[Dict]":
        """Creates a new data stream from a path to a file with JSON array, where each item is a
        valid JSON (python dict)

        Args:
            filename (str): A path to a utf8 encode text file with JSON array,
                where each item is a valid JSON (python dict)

        Returns:
            DataStream: A new data stream that produces python dict items each
                containing a single JSON object specified by 'filename'

        Notes:
            This class method returns a data stream over the valid JSON objects of a single
            JSON array text file.

        Examples:
            For a JSON file that looks like:
                [
                { a: 1, b: 2, c: False },
                { a: 2, b: 3 },
                { a: 3, c: True }
                ]

            >>> json_data_stream = DataStream.from_json_array('sample.json')
            >>> for data_item in json_data_stream:
            >>>     print(data_item)
            { a: 1, b: 2, c: False }
            { a: 2, b: 3 }
            { a: 3, c: True }

        """
        error.file_check("<COR39609575E>", filename)

        return cls(cls._from_json_array_generator, filename)

    @classmethod
    def _from_json_array_generator(cls, filename):
        # open the file
        with open(filename, mode="rb") as json_fh:
            log.debug2("Loading JSON array file: %s", filename)

            # for each {} object of the array
            try:
                for item_idx, obj in enumerate(ijson.items(json_fh, "item")):
                    log.debug2("Loading object index %d", item_idx)
                    yield obj

            except ijson.JSONError:
                error(
                    "<COR85596551E>",
                    ValueError("Invalid JSON object in `{}`".format(filename)),
                )

    @classmethod
    def from_csv(cls, filename: str, *args, skip=0, **kwargs) -> "DataStream[List]":
        """Create a new data stream from a csv (comma separated value) file where each data item
        corresponds to a line of the csv file and consists of a list containing the comma separated
        values.

        Args:
            filename (str): A path to a csv file that has rows corresponding to
                data items and columns corresponding to the elements of each
                data item.
            skip (int): Number of lines to skip at the beginning of the csv
                file.  This is often useful for skipping a header line.
            args, kwargs: Additional arguments passed to the `csv.reader` function.
                These can be used to specify the delimiter or other csv settings.
        Returns:
            DataStream: A data stream that produces a data item for each line of
                the csv file and where each element of the data item corresponds
                to a column in the csv file.Examples:
            For a sample.csv that looks like:
                a, b, c
                d, e, f
            >>> csv_stream = DataStream.from_csv('sample.csv')
            >>> for data_item in csv_stream:
            >>>     print(data_item)
            ['a', 'b', 'c']
            ['d', 'e', 'f']
        """
        # verify that the csv file exists and is a regular file
        if not os.path.exists(filename) or not os.path.isfile(filename):
            error(
                "<COR82308234E>",
                FileNotFoundError(
                    "csv filename `{}` does not exist or is not a regular file.".format(
                        filename
                    )
                ),
            )

        return cls(cls._from_csv_generator, filename, skip, *args, **kwargs)

    @classmethod
    def _from_csv_generator(cls, filename, skip, *csv_args, **csv_kwargs):
        # open the csv file (closure around `filename`)
        with open(filename, mode="r", encoding="utf8") as fh:
            # skip lines if requested
            for _ in range(skip):
                # pylint: disable=stop-iteration-return
                next(fh)

            # for each line of the csv file, yield a list
            for line in csv.reader(fh, *csv_args, **csv_kwargs):
                yield line

    @classmethod
    def from_header_csv(cls, filename: str, *args, **kwargs) -> "DataStream[Dict]":
        """Create a new data stream from a csv where the first row is a header
        and each subsequent row is an element. The yielded elements are tuples
        of dicts where each dict pairs the row values with the corresponding
        column headers.

        Args:
            filename (str): A path to a csv file that has rows corresponding to
                data items and columns corresponding to the elements of each
                data item.
            args, kwargs: Additional arguments passed to the `csv.reader` function.
                These can be used to specify the delimiter or other csv settings.
        Returns:
            DataStream: A data stream that produces a data item for each line of
                the csv file and where each element of the stream is a dict
                representation of the fieldsExamples:
            For a sample.csv that looks like:
                foo, bar, baz
                a, b, c
                d, e, f
            >>> csv_stream = DataStream.from_csv('sample.csv')
            >>> for data_item in csv_stream:
            >>>     print(data_item)
            {"foo": "a", "bar": "b", "baz": "c"}
            {"foo": "d", "bar": "e", "baz": "f"}
        """
        # verify that the csv file exists and is a regular file
        if not os.path.exists(filename) or not os.path.isfile(filename):
            error(
                "<COR44308234E>",
                FileNotFoundError(
                    "csv filename `{}` does not exist or is not a regular file.".format(
                        filename
                    )
                ),
            )

        return cls(cls._from_header_csv_generator, filename, *args, **kwargs)

    @classmethod
    def _from_header_csv_generator(cls, filename, *csv_args, **csv_kwargs):
        # open the csv file (closure around `filename`)
        with open(filename, mode="r", encoding="utf8") as fh:

            # for each line of the csv file, yield a dict
            for line in csv.DictReader(fh, *csv_args, **csv_kwargs):
                yield line

    @classmethod
    def from_txt(cls, filename: str) -> "DataStream[str]":
        """Create a new data stream from a path to a utf8 encoded text file where each data item
        corresponds to a single line of the file.

        Args:
            filename (str): A path to a utf8 encode text file with each line
                corresponding to a data item.

        Returns:
            DataStream: A new data stream that produces string data items each
                containing a single line from the file specified by `filename`.

        Notes:
            This class method returns a data stream over the lines of a single text file.  In
            order to construct a datastream over separate files, rather than lines, consider using
            `.from_txt_collection`.

        Examples:
            For a text file that looks like:
                first line
                second line
                third line

            >>> txt_line_stream = DataStream.from_file('sample.txt')
            >>> for data_item in txt_line_stream:
            >>>     print(data_item)
            first line
            second line
            third line
        """
        error.file_check("<COR79693043E>", filename)

        return cls(cls._from_txt_generator, filename)

    @classmethod
    def _from_txt_generator(cls, filename):
        # open the file (closure around `filename`)
        with open(filename, mode="r", encoding="utf8") as fh:
            # for each line of the file
            for line in fh:
                # strip new lines and carriage returns and yield the line
                yield line.rstrip("\n\r")

    @classmethod
    def from_file(cls, filename: str) -> "DataStream[Union[Dict, Tuple, str]]":
        """Loads up a DataStream from a file.
            Will call the correct DataStream.from_caikit static constructor based on the
            file extension

            The data items returned in the data stream are:
            For JSON:
                dictionaries
            For all other files (besides CSV for now)
                strings (1 per line)
        Args:
            filename (str): name of file

        Returns:
            DataStream: resulting datastream from file
        """
        # file detection
        _, file_ext = os.path.splitext(filename)
        # choose the right from_* fn
        if file_ext.lower() == ".json":
            log.debug2("Detected .json extension, loading %s as a json file", filename)
            return DataStream.from_json_array(filename)

        if file_ext.lower() == ".csv":
            log.debug2("Detected .csv extension, loading %s as a csv file", filename)
            return DataStream.from_csv(filename)

        log.debug2("Loading %s as a raw text file", filename)
        # TODO: test this at some point (this path is unused currently)
        return DataStream.from_txt(filename)

    @classmethod
    def _from_collection(
        cls, dirname: str, extension: str, file_opener
    ) -> "DataStream[Union[Dict, Tuple, str]]":
        """Create a new data stream from a path containing multiple files where
        each data item corresponds to the entire serialized content in a single file. The
        file_handler function does the serialization of individual files

        Args:
            dirname (str): A directory path containing a number of utf8 encoded
                text files with the `.txt` filename extension.
            extension (str): Extension of the file. Note that all files are read
                in the same utf8 encoding.
            file_opener (function): Function to deserialize a file on disk to
                memory

        Returns:
            DataStream: A new data stream that produces string data items each
                containing the text contained in a single file found in
                `dirname`.

        Notes:
            Each data item in this data stream represents the *entire* text contained in a single
            file and are not split by line or otherwise.
        """
        # verify that `dirname` exists
        cls._verify_dir(dirname)

        return cls(cls._from_collection_generator, dirname, extension, file_opener)

    @classmethod
    def _from_collection_generator(cls, dirname, extension, file_opener):
        # glob `*.txt` files in `dirname` (closure around `dirname`)
        for filename in glob(os.path.join(dirname, "*." + extension)):
            yield file_opener(filename)

    @classmethod
    def from_txt_collection(cls, dirname: str, extension="txt") -> "DataStream[str]":
        """Create a new data stream from a path containing multiple utf8 encoded text files where
        each data item corresponds to the entire text contained in a single file.

        Args:
            dirname (str): A directory path containing a number of utf8 encoded
                text files with the `.txt` filename extension.
            extension: str (Optional)
                Optional extension of the text file. Note that all files are read in the same
                utf8 encoding. Defaults to 'txt'

        Returns:
            DataStream: A new data stream that produces string data items each
                containing the text contained in a single `.txt` (or specified
                extension) file found in `dirname`.

        Notes:
            Each data item in this data stream represents the *entire* text contained in a single
            file and are not split by line or otherwise.
        """
        return cls._from_collection(dirname, extension, fileio.load_txt)

    @classmethod
    def from_json_collection(
        cls, dirname: str, extension="json"
    ) -> "DataStream[Union[Dict, Tuple, List]]":
        """Create a new data stream from a path containing multiple JSON files where
        each data item corresponds to the entire serialized JSON contained in a single file.

        Args:
            dirname (str): A directory path containing a number of utf8 encoded
                text files with the `.txt` filename extension.
            extension: str (Optional)
                Optional extension of the JSON file. Note that all files are read in the same
                utf8 encoding. Defaults to 'json'

        Returns:
            DataStream: A new data stream that produces string data items each
                containing the text contained in a single `.json` (or specified
                extension) file found in `dirname`.

        Notes:
            Each data item in this data stream represents the *entire* text contained in a single
            file and are not split by line or otherwise.
        """
        return cls._from_collection(dirname, extension, fileio.load_json)

    @classmethod
    def from_csv_collection(cls, dirname: str) -> "DataStream[Dict]":
        """Create a new data stream by chaining data streams from each of the file from a path
        containing multiple csv files where each file can have 1 or more data item.

        Args:
            dirname (str): A directory path containing a number of csv files

        Returns:
            DataStream: A new data stream that is chained from all data streams
                by reading (from_header_csv) all files in all `.csv` files found
                in `dirname`. All data items are dicts.
        """
        # verify that `dirname` exists
        cls._verify_dir(dirname)

        return cls(cls._from_csv_collection_generator, dirname)

    @classmethod
    def _from_csv_collection_generator(cls, dirname):
        # list of data_streams created from different files
        data_stream_list = []
        # glob `*.txt` files in `dirname` (closure around `dirname`)
        for filename in glob(os.path.join(dirname, "*.csv")):
            data_stream_list.append(cls.from_header_csv(filename=filename))
        # yield the combined data item once flattened
        for data_item in DataStream.chain(data_stream_list).flatten():
            yield data_item

    @classmethod
    def from_jsonl_collection(cls, dirname: str) -> "DataStream[Dict]":
        """Create a new data stream by chaining data streams from each of the file from a path
        containing multiple jsonl files where each file can have 1 or more data item.

        Args:
            dirname (str): A directory path containing a number of jsonl files

        Returns:
            DataStream: A new data stream that is chained from all data streams
                by reading (from_jsonl) all files in all `.jsonl` files found in
                `dirname`.
        """
        # verify that `dirname` exists
        cls._verify_dir(dirname)

        return cls(cls._from_jsonl_collection_generator, dirname)

    @classmethod
    def _from_jsonl_collection_generator(cls, dirname):
        # list of data_streams created from different files
        data_stream_list = []
        # glob `*.txt` files in `dirname` (closure around `dirname`)
        for filename in glob(os.path.join(dirname, "*.jsonl")):
            data_stream_list.append(cls.from_jsonl(filename=filename))
        # yield the combined data item once flattened
        for data_item in DataStream.chain(data_stream_list).flatten():
            yield data_item

    def train_test_split(
        self, test_split=0.25, seed=None
    ) -> "Tuple[DataStream[T], DataStream[T]]":
        """Split the current datastream into train/test substreams.

        Args:
            test_split (float): The fraction of examples to assign to the test
                substream, in [0, 1]
            seed (int | None): The seed for initializing the random assignment.
                If not provided, a randomly chosen seed will be used.

        Returns:
            tuple(DataStream, DataStream): Two substreams: a train set
                substream, and a test set substream
        """
        assert 0.0 <= test_split <= 1.0

        if seed is None:
            seed = random.randint(0, 10000)

        def train_generator_func():
            rng = random.Random(seed)
            for data_item in self:
                if rng.random() > test_split:
                    yield data_item

        def test_generator_func():
            rng = random.Random(seed)
            for data_item in self:
                if rng.random() <= test_split:
                    yield data_item

        return DataStream(train_generator_func), DataStream(test_generator_func)

    # pylint: disable=no-self-argument
    def chain(*args) -> "DataStream":
        """Chain multiple data streams together sequentially.  The returned data stream produces
        the data items from each passed data stream in turn.

        Args:
            args (tuple(DataStream)): A tuple containing the data streams to
                chain, passed as variadic arguments.

        Returns:
            DataStream: A new data stream that produces the data items from the
                provided data streams sequentially.
        """
        return DataStream(lambda: itertools.chain(*args))

    # pylint: disable=keyword-arg-before-vararg
    def filter(
        self, func=lambda data_item: data_item, *args, **kwargs
    ) -> "DataStream[T]":
        """Skip elements in the data stream as identified by a passed function.

        Args:
            func (callable(data_item)): The function used to identify data items
                that will be filtered.  The function takes a single data item as
                an argument and returns `True` in order to keep the element and
                `False` in order to skip it.  The default filter function
                removes falsey values.

        Returns:
            DataStream: A new data stream that produces the data items from the
                current data stream only when `func` evaluates to true.
        """
        error.value_check(
            "<COR59884427E>", callable(func), "filter function is not callable"
        )

        return DataStream(
            lambda: (
                data_item for data_item in self if func(data_item, *args, **kwargs)
            )
        )

    def shuffle(self, buffer_size, seed=None) -> "DataStream[T]":
        """Randomly shuffles the elements of this dataset. If buffer_size is smaller than the full
        size of the full data stream, it is a partial random shuffle which is similar to
        Tensorflow's dataset shuffle. For instance, if your dataset contains 10,000 elements but
        buffer_size is set to 1,000, then shuffle will initially select a random element from only
        the first 1,000 elements in the buffer. Once an element is selected, its space in the
        buffer is replaced by the next (i.e. 1,001-st) element, maintaining the 1,000 element
        buffer.

        Args:
            buffer_size (int): the size of the buffer space, should be greater
                than 0
            seed (int | None): The seed for initializing the random assignment.
                If not provided, a randomly chosen seed will be used.

        Returns:
            DataStream: A new data stream after shuffled.
        """
        # make sure buffer space is valid
        error.type_check("<COR06395206E>", int, buffer_size=buffer_size)
        error.value_check(
            "<COR78471251E>", buffer_size > 0, "Buffer size must be an int > 0"
        )

        if seed is None:
            seed = random.randint(0, 10000)

        def generator_func():
            buffer = []
            random.seed(seed)
            if self._length is not None and self._length <= buffer_size:
                buffer = list(itertools.islice(self, buffer_size))
            else:
                count = 0
                for e in self:
                    if count < buffer_size:
                        buffer.append(e)
                        count += 1
                    else:
                        idx = random.randint(0, buffer_size - 1)
                        item = buffer[idx]
                        buffer[idx] = e
                        yield item

            random.shuffle(buffer)
            for item in buffer:
                yield item

        return DataStream(generator_func)

    def eager(self) -> "DataStream[T]":
        """Evaluate the data stream, place it into memory and return a new data stream over these
        static values.  This is useful if your data stream can fit into memory, at least up to a
        certain point, and it will not be efficient to lazily and, potentially, re-evaluate the
        stream each time it is iterated over.

        Returns:
            DataStream: A new data stream that iterates over the evaluated, in-
                memory data items in this stream.
        """
        return DataStream.from_iterable(list(self))

    def map(self, func, *args, **kwargs) -> "DataStream":
        """Apply a function to each element in the data stream.

        Args:
            func (callable(*args, **kwargs)): A function this is lazily applied
                to each element in the data stream.
            *args, **kwargs
                Additional arguments to pass `func`.

        Returns:
            DataStream: A new data stream with `func` applied to each element.
        """
        return DataStream(
            lambda: (func(data_item, *args, **kwargs) for data_item in self)
        )

    def flatten(self) -> "DataStream":
        """Convert a 2-level nested stream into a flattened stream

        Returns:
            DataStream: A new data stream with inner stream items 'flattened'
        """

        def generator_func():
            for inner_stream in self:
                for data_item in inner_stream:
                    yield data_item

        return DataStream(generator_func)

    # pylint: disable=no-self-argument
    def zip(*args) -> "DataStream":
        """Combine the data items of multiple data streams together in tuples.

        Args:
            args (tuple(DataStream)): A tuple containing the data streams to be
                zip, passed as variadic arguments.

        Returns:
            DataStream: A data stream that produces the zipped data items.

        Notes:
            A `ValueError` is raised when the stream is iterated over if any of the zipped data
            streams do not have the same length.  Since streams are evaluated lazily, however, this
            error condition will only be detected and raised when the stream is being iterated over.
        """
        error.type_check_all("<COR19533030E>", DataStream, args=args)

        def generator_func():
            # create a unique object as a sentinel
            sentinel = object()

            # zip the data items together and pad with the sentinel
            for zipped_data_items in itertools.zip_longest(*args, fillvalue=sentinel):
                # if the sentinel is detected, the data streams do not have the same length
                if sentinel in zipped_data_items:
                    error(
                        "<COR83794589E>",
                        ValueError(
                            "Failed to zip data streams with different lengths."
                        ),
                    )

                yield zipped_data_items

        return DataStream(generator_func)

    def peek(self) -> T:
        """Returns the first element of the stream, or raises IndexError if stream is empty"""
        try:
            return next(iter(self))
        except StopIteration:
            error.log_raise("<COR48484123E>", IndexError("Cannot peek empty stream"))

    def augment(
        self,
        augmentor,
        aug_cycles,
        *,
        post_augment_func=None,
        augment_index=None,
        enforce_determinism=True,
    ) -> "DataStream[T]":
        error.type_check("<COR45851623E>", int, aug_cycles=aug_cycles)
        error.type_check("<COR80001982E>", AugmentorBase, augmentor=augmentor)
        error.type_check(
            "<COR87701982E>", bool, enforce_determinism=enforce_determinism
        )
        error.value_check(
            "<COR56795914E>", aug_cycles > 0, "Augmentation cycles must be an int > 0"
        )
        if post_augment_func is not None and not callable(post_augment_func):
            error(
                "<COR32996115E>",
                TypeError("Post augmentation operation is not callable"),
            )
        contains_iterables = isinstance(self.peek(), (tuple, list))
        # Explode if we have an augment index that is not applicable (i.e., don't have lists or
        # tuple objects being considered to apply it against).
        if not contains_iterables and augment_index is not None:
            error(
                "<COR31116115E>",
                ValueError(
                    "augment_index cannot be used unless stream contains lists/tuples"
                ),
            )
        # Explode if we don't have an augmentation index, but we need one (i.e., we have lists or
        # tuple objects, but don't know how to use the augmentor)
        if contains_iterables and augment_index is None:
            error(
                "<COR31316445E>",
                ValueError(
                    "augment_index must be given to augment a stream of lists/tuples"
                ),
            )

        def generator_func():
            for cycle_num in range(aug_cycles + 1):
                for obj in self:
                    if (
                        contains_iterables
                        and augment_index
                        and len(obj) <= augment_index
                    ):
                        error(
                            "<COR31352545E>",
                            IndexError(
                                "augment_index is out of bounds of obj in stream"
                            ),
                        )

                    # Figure out what we need to apply the augmentor to, then apply it. After
                    # that, apply the post augmentation func if one is provided.
                    augmentable = obj if augment_index is None else obj[augment_index]
                    # If this is the first cycle, don't apply the augmentor, otherwise do it.
                    augmented = (
                        augmentor.augment(augmentable) if cycle_num else augmentable
                    )
                    # In some special cases, we've designed things following the augmentor pattern
                    # that may return None for some inputs. If the augmentor is designed for this,
                    # i.e., sets .produces_none=True in the subclass, filter these objects out
                    # from the returned stream. Note that this WILL drop None values from the
                    # original dataset at the moment as well, as keeping None in the input stream
                    # is presumably a rare behavior.
                    if augmented is None and not augmentor.produces_none:
                        error(
                            "<COR34377515E>",
                            ValueError("Augmentor produced [None] unexpectedly"),
                        )
                    elif augmented is None:
                        continue

                    if post_augment_func is not None:
                        augmented = post_augment_func(augmented)
                    # If there is no augmentation index, we're done - return the augmented object
                    if augment_index is None:
                        yield augmented
                    # Otherwise we need to repack the augmented object back into the tuple,
                    # where it lives in peace with everything else in this item.
                    else:
                        yield tuple(
                            elem if idx != augment_index else augmented
                            for idx, elem in enumerate(obj)
                        )
            # Reset the augmentor after all cycles to ensure the DataStream is deterministic
            if enforce_determinism:
                augmentor.reset()

        return DataStream(generator_func)

    def __add__(self, other):
        """The addition operator for data streams is equivalent to calling `.chain` and combines
        this data stream with another sequentially.
        """
        return self.chain(other)

    def __getitem__(self, idx) -> T:
        """Index or slice each data item.  This is valuable for creating new data streams over the
        elements of a stream that produces tuples, lists, arrays, et cetra.

        Args:
            idx (int or slice): The index or slice to be applied to each data
                item.

        Returns:
            DataStream: A new data stream with `data_item[idx]` applied to each
                data item.

        Notes:
            This operation may be somewhat counter intuitive since `data_stream[0]` does not return
            the first element of the data stream and, instead, returns a new data stream that
            produces `data_item[0]` for each data item.

            This operation may fail with a `TypeError` if the data items in the stream are not
            subscriptable.
        """
        return DataStream(lambda: (data_item[idx] for data_item in self))

    def __iter__(self):
        """Return an iterator or generator over all of the data items in this data stream.  Data
        streams are reentrant in the sense that they can be iterated over multiple times.
        """
        # call the generator function to create an iterable yielding the data items
        # pass in the variadic arguments saved during construction of the data stream
        generator = self.generator_func(*self.generator_args, **self.generator_kwargs)
        if not isinstance(generator, collections.abc.Iterable):
            error(
                "<COR35849950E>",
                RuntimeError("`generator_func` did not return an iterable"),
            )

        return generator

    def __len__(self):
        """Return the number of data items contained in this data stream.  This requires that the
        data stream be iterated over, which may be time consuming.  This value is then stored
        internally so that subsequent calls do not iterate over the data stream again.
        """
        if self._length is None:
            self._length = sum(1 for data_item in self)

        return self._length

    def __or__(self, module):
        """Feed this data stream into the `.stream` method of a module.  This is syntactic sugar
        that allows modules to be chained like `data_stream | module1 | module2` rather than the
        equivalent `module2.stream(module1.stream(data_stream))`.
        """
        return module.stream(self)

    # Helper functions

    @staticmethod
    def _verify_dir(dirname):
        # verify that `dirname` exists
        if not os.path.exists(dirname):
            error(
                "<COR82306771E>",
                FileNotFoundError(
                    "Could not find collection directory `{}`".format(dirname)
                ),
            )

        # verify that `dirname` is a directory
        if not os.path.isdir(dirname):
            error(
                "<COR82306849E>",
                NotADirectoryError(
                    "collection path `{}` is not a directory".format(dirname)
                ),
            )
