# Standard
from dataclasses import dataclass
from typing import List
import json
import os
import pickle
import tempfile

# Third Party
import pytest

# Local
from caikit.core.data_model import DataObjectBase, dataobject
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.interfaces.common.data_model.stream_sources import S3Files
from caikit.runtime.service_generation.data_stream_source import (
    DataStreamSourceBase,
    _make_data_stream_source_type_name,
    make_data_stream_source,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from sample_lib.data_model.sample import SampleTrainingType
from tests.conftest import temp_config
import caikit

################################################
# Fixtures


@pytest.fixture
def sample_jsonl_file() -> str:
    jsonl_content = []
    jsonl_content.append(json.dumps({"number": 1}))
    jsonl_content.append(json.dumps({"number": 2}))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as handle:
        for row in jsonl_content:
            handle.write(row)
            handle.write("\n")
        handle.flush()
        yield handle.name


@pytest.fixture
def sample_json_content() -> List:
    json_content = []
    json_content.append(json.dumps({"number": 0}))
    json_content.append(json.dumps({"number": 1}))
    json_content.append(json.dumps({"number": 2}))
    return json_content


@pytest.fixture
def sample_json_dir(sample_json_content) -> str:
    with tempfile.TemporaryDirectory() as tempdir:
        for index, data in enumerate(sample_json_content):
            fname = os.path.join(tempdir, f"sample_{index}.json")
            with open(fname, "w") as handle:
                handle.write(data)
        yield tempdir


@pytest.fixture
def sample_jsonl_dir(sample_json_content) -> str:
    with tempfile.TemporaryDirectory() as tempdir:
        # Add all content to all files, different from what
        # we have in sample_json_dir above
        for index, _ in enumerate(sample_json_content):
            fname = os.path.join(tempdir, f"sample_{index}.jsonl")
            with open(fname, "w") as handle:
                for row in sample_json_content:
                    handle.write(row)
                    handle.write("\n")
        yield tempdir


@pytest.fixture
def sample_csv_dir() -> str:
    csv_header = "number"
    csv_content = []
    csv_content.append("0")
    csv_content.append("1")
    csv_content.append("2")
    with tempfile.TemporaryDirectory() as tempdir:
        for index, data in enumerate(csv_content):
            fname = os.path.join(tempdir, f"sample_{index}.csv")
            with open(fname, "w") as handle:
                handle.write(csv_header)
                handle.write("\n")
                handle.write(data)
                handle.write("\n")
        yield tempdir


################################################
## Helper functions


def validate_data_stream(data_stream, length, data_item_type, data_item_length=None):
    assert isinstance(data_stream, DataStream)
    assert len(data_stream) == length
    assert sum(1 for _ in data_stream) == length

    for data_item in data_stream:
        assert isinstance(data_item, data_item_type)
        if data_item_length is not None:
            assert len(data_item) == data_item_length


################################################
# Normal Tests


def test_make_data_stream_source_type_name():
    assert "DataStreamSourceInt" == _make_data_stream_source_type_name(int)
    assert "DataStreamSourceFloat" == _make_data_stream_source_type_name(float)
    assert "DataStreamSourceStr" == _make_data_stream_source_type_name(str)
    assert "DataStreamSourceBool" == _make_data_stream_source_type_name(bool)
    assert "DataStreamSourceSampleTrainingType" == _make_data_stream_source_type_name(
        SampleTrainingType
    )


def test_multiple_make_data_stream_source():
    """Defining 2 data stream sources in any order should not fail"""

    # Create 2 data stream sources
    make_data_stream_source(int)
    make_data_stream_source(SampleTrainingType)

    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    ds = stream_type(jsondata=stream_type.JsonData(data=[1, 2, 3]))
    proto_repr = ds.to_proto()
    assert ds.from_proto(proto_repr).to_proto() == proto_repr
    assert stream_type.from_proto(proto_repr).to_proto() == proto_repr


def test_data_model_element_type(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    assert isinstance(stream_type._to_element_type({"number": 1}), SampleTrainingType)


def test_primitive_element_type(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    assert isinstance(stream_type._to_element_type(1), int)


def test_make_data_stream_source_types():
    assert issubclass(make_data_stream_source(int), DataStreamSourceBase)
    assert issubclass(make_data_stream_source(float), DataStreamSourceBase)
    assert issubclass(make_data_stream_source(str), DataStreamSourceBase)


def test_make_data_stream_source_empty(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type()
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 0, None)


def test_pickle_round_trip_primitive():
    """Make sure that a source wrapping a primitive type can be round-tripped
    cleanly with pickle
    """
    stream_source = make_data_stream_source(float)
    inst = stream_source(jsondata=stream_source.JsonData(data=[1.23, -2.34]))
    round_trip = pickle.loads(pickle.dumps(inst))
    assert round_trip.to_dict() == inst.to_dict()


def test_pickle_round_trip_data_model():
    """Make sure that a source wrapping a data model object can be round-tripped
    cleanly with pickle
    """

    @dataobject
    class Foo(DataObjectBase):
        foo: int
        bar: str

    stream_type = make_data_stream_source(Foo)
    inst = stream_type(
        jsondata=stream_type.JsonData(data=[Foo(1, "two"), Foo(2, "three")])
    )
    round_trip = pickle.loads(pickle.dumps(inst))
    assert round_trip.to_dict() == inst.to_dict()


def test_data_stream_source_as_data_stream():
    """Make sure that a DataStreamSource works exactly like a DataStream"""
    stream_source = make_data_stream_source(float)
    source_list = [1.23, -2.34]

    # Test inline jsondata
    inst = stream_source(jsondata=stream_source.JsonData(data=source_list))
    assert list(inst) == source_list

    # Test from file
    # NOTE: Plain JSON doesn't round trip because ijson returns a special
    #   Decimal wrapper around raw values
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as handle:
        handle.write("\n".join([str(val) for val in source_list]))
        handle.flush()
        inst = stream_source(file=stream_source.File(filename=handle.name))
        assert list(inst) == source_list


def test_data_stream_source_base_path():
    """Make sure that a globally configured base path is used"""
    stream_source = make_data_stream_source(int)
    source_data = [1, 2, 3, 4]
    with tempfile.TemporaryDirectory() as workdir:
        with temp_config({"data_streams": {"file_source_base": workdir}}):
            nested_dir = os.path.join("foo", "bar")
            full_nested_dir = os.path.join(workdir, nested_dir)
            os.makedirs(full_nested_dir)
            fname = os.path.join(nested_dir, "data.json")
            full_fname = os.path.join(workdir, fname)
            with open(full_fname, "w") as handle:
                handle.write(json.dumps(source_data))

            # Make sure it works with the relative file path
            assert (
                list(stream_source(file=stream_source.File(filename=fname)))
                == source_data
            )

            # Make sure it works with the absolute file path
            assert (
                list(stream_source(file=stream_source.File(filename=full_fname)))
                == source_data
            )

            # Make sure it works with the relative directory
            assert list(
                stream_source(
                    directory=stream_source.Directory(
                        dirname=nested_dir,
                        extension="json",
                    )
                )
            ) == [source_data]

            # Make sure it works with the absolute directory
            assert list(
                stream_source(
                    directory=stream_source.Directory(
                        dirname=full_nested_dir,
                        extension="json",
                    )
                )
            ) == [source_data]

            # Make sure bad paths still raise the right errors
            with pytest.raises(CaikitRuntimeException):
                list(
                    stream_source(file=stream_source.File(filename="invalid/path.json"))
                )
            with pytest.raises(CaikitRuntimeException):
                list(
                    stream_source(
                        directory=stream_source.Directory(dirname="invalid/path")
                    )
                )


#################
# JSON data
#################


def test_make_data_stream_source_jsondata(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


def test_make_data_stream_source_jsondata_other_task(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    ds = stream_type(jsondata=stream_type.JsonData(data=[1]))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 1, int)


#################
# File
#################


def test_make_data_stream_source_jsonfile(sample_train_service, sample_json_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=stream_type.File(filename=sample_json_file))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


def test_make_data_stream_source_csvfile(sample_train_service, sample_csv_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=stream_type.File(filename=sample_csv_file))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


def test_make_data_stream_source_jsonlfile(sample_train_service, sample_jsonl_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=stream_type.File(filename=sample_jsonl_file))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


#################
# List of files
#################


def test_make_data_stream_source_list_of_json_files(
    sample_train_service, sample_json_file
):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    # does NOT work with sample_json_dir as each file needs to have data in array
    ds = stream_type(
        listoffiles=stream_type.ListOfFiles(files=[sample_json_file, sample_json_file])
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 4, SampleTrainingType)


def test_make_data_stream_source_list_of_csv_files(
    sample_train_service, sample_csv_dir
):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    sample_files = [
        os.path.join(sample_csv_dir, file) for file in os.listdir(sample_csv_dir)
    ]
    # send all files but last one
    ds = stream_type(listoffiles=stream_type.ListOfFiles(files=sample_files[:-1]))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


def test_make_data_stream_source_list_of_jsonl_files(
    sample_train_service, sample_jsonl_dir
):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    sample_files = [
        os.path.join(sample_jsonl_dir, file) for file in os.listdir(sample_jsonl_dir)
    ]
    # send all files but last one
    ds = stream_type(listoffiles=stream_type.ListOfFiles(files=sample_files[:-1]))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 6, SampleTrainingType)


#################
# Directories
#################


def test_make_data_stream_source_jsondir(sample_train_service, sample_json_dir):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(directory=stream_type.Directory(dirname=sample_json_dir))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 3, SampleTrainingType)


def test_make_data_stream_source_csvdir(sample_train_service, sample_csv_dir):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        directory=stream_type.Directory(dirname=sample_csv_dir, extension="csv")
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 3, SampleTrainingType)


def test_make_data_stream_source_jsonldir(sample_train_service, sample_jsonl_dir):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        directory=stream_type.Directory(dirname=sample_jsonl_dir, extension="jsonl")
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 9, SampleTrainingType)


def test_data_stream_operators():
    """Make sure that standard DataStream operators work as expected"""
    stream_type = make_data_stream_source(int)
    ds = stream_type(jsondata=stream_type.JsonData(data=[1, 2, 3, 4]))
    train, test = ds.train_test_split()
    assert sorted(list(train) + list(test)) == [1, 2, 3, 4]
    ds2 = DataStream.from_iterable([5, 6, 7])
    assert list(ds.chain(ds2)) == [1, 2, 3, 4, 5, 6, 7]
    assert list(ds.filter(lambda x: x % 2)) == [1, 3]
    assert len(list(ds.shuffle(2))) == 4
    assert list(ds.map(lambda x: x * 2)) == [2, 4, 6, 8]
    assert list(DataStream.from_iterable([ds]).flatten()) == [1, 2, 3, 4]
    zipped_ds = ds.zip(DataStream.from_iterable([5, 6, 7, 8]))
    assert list(zipped_ds) == [
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
    ]
    assert list(zipped_ds[0]) == [1, 2, 3, 4]


################################################
# Error Tests


def test_make_data_stream_source_invalid_file_raises(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=stream_type.File(filename="invalid_file"))
    with pytest.raises(CaikitRuntimeException):
        ds.to_data_stream()


def test_make_data_stream_source_invalid_dir_raises(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(directory=stream_type.Directory(dirname="invalid_dir"))
    with pytest.raises(CaikitRuntimeException):
        ds.to_data_stream()


def test_data_stream_source_single_oneof(sample_train_service):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    with pytest.raises(CaikitRuntimeException):
        stream_type(
            jsondata=stream_type.JsonData(data=[SampleTrainingType(1)]),
            directory=stream_type.Directory(dirname="invalid_dir"),
        )


def test_make_data_stream_source_invalid_ext_dir(sample_train_service):
    with tempfile.TemporaryDirectory() as tempdir:
        fname = os.path.join(tempdir, f"sample.txt")
        with open(fname, "w") as handle:
            handle.write("something")
        stream_type = (
            caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
        )
        ds = stream_type(
            directory=stream_type.Directory(dirname=tempdir, extension="txt")
        )
        with pytest.raises(CaikitRuntimeException) as e:
            ds.to_data_stream()
        assert "Extension not supported!" in e.value.message


def test_make_data_stream_source_no_files_w_ext_dir(
    sample_train_service, sample_jsonl_dir
):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        directory=stream_type.Directory(dirname=sample_jsonl_dir, extension="csv")
    )
    with pytest.raises(CaikitRuntimeException) as e:
        ds.to_data_stream()
    assert "contains no source files with extension" in e.value.message


def test_s3_not_implemented(sample_train_service, sample_jsonl_dir):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(s3files=S3Files())
    # Explicit .to_data_stream will fail
    with pytest.raises(
        NotImplementedError,
        match="S3Files are not implemented as stream sources in this runtime.",
    ) as e:
        ds.to_data_stream()

    # And so would iterating on the data stream source directly
    with pytest.raises(
        NotImplementedError,
        match="S3Files are not implemented as stream sources in this runtime.",
    ) as e:
        for val in ds:
            _ = val
