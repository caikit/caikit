# Standard
import json
import os
import pickle
import shutil
import tempfile

# Third Party
import pytest

# Local
from caikit.core.data_model import DataObjectBase, dataobject
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    FileReference,
    S3Files,
)
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
## Helper functions


@pytest.fixture(autouse=True)
def train_service(sample_train_service):
    """This autoused fixture ensures that the training APIs will be created
    when individual tests are run. Each test, however, does not
    use the fixture explicitly
    """
    pass


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


def test_make_data_stream_source_types():
    assert issubclass(make_data_stream_source(int), DataStreamSourceBase)
    assert issubclass(make_data_stream_source(float), DataStreamSourceBase)
    assert issubclass(make_data_stream_source(str), DataStreamSourceBase)


def test_make_data_stream_source_empty():
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

    validate_data_stream(round_trip, 2, float)


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

    validate_data_stream(round_trip, 2, Foo)


def test_pickle_round_trip_file(sample_json_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    data_stream = stream_type(file=FileReference(filename=sample_json_file))

    round_trip = pickle.loads(pickle.dumps(data_stream))

    validate_data_stream(round_trip, 2, SampleTrainingType)


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
        inst = stream_source(file=stream_source.FileReference(filename=handle.name))
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
                list(stream_source(file=stream_source.FileReference(filename=fname)))
                == source_data
            )

            # Make sure it works with the absolute file path
            assert (
                list(
                    stream_source(file=stream_source.FileReference(filename=full_fname))
                )
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
                    stream_source(
                        file=stream_source.FileReference(filename="invalid/path.json")
                    )
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


def test_make_data_stream_source_jsondata():
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


def test_make_data_stream_source_jsondata_other_task():
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    ds = stream_type(jsondata=stream_type.JsonData(data=[1]))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 1, int)


#################
# File
#################


def test_make_data_stream_source_jsonfile(sample_json_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=FileReference(filename=sample_json_file))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    print(ds, data_stream)
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


def test_make_data_stream_source_csvfile(sample_csv_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=FileReference(filename=sample_csv_file))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 3, SampleTrainingType)


def test_make_data_stream_source_jsonlfile(sample_jsonl_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=FileReference(filename=sample_jsonl_file))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 4, SampleTrainingType)


def test_make_data_stream_source_jsonlfile_extra_fields(tmp_path):
    """Test that extra fields that may be present in files are just ignored"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    file_path = os.path.join(str(tmp_path), "extra_fields.jsonl")
    with open(file_path, "w") as fp:
        fp.write(
            """
        {"number": 2, "label": "bar", "description": "bar"}
        {"number": 3, "label": "foo", "description": "foo"}
        """
        )

    ds = stream_type(file=FileReference(filename=file_path))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 2, SampleTrainingType)


def test_make_data_stream_source_from_file_with_no_extension(
    sample_json_file, sample_jsonl_file, sample_csv_file, tmp_path
):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    for file in [sample_json_file, sample_jsonl_file, sample_csv_file]:
        no_extension_filename = os.path.join(str(tmp_path), "no_extension")
        shutil.copyfile(file, no_extension_filename)

        ds = stream_type(file=FileReference(filename=no_extension_filename))
        assert isinstance(ds, DataStreamSourceBase)

        data_stream = ds.to_data_stream()
        assert isinstance(data_stream, DataStream)

        assert isinstance(data_stream.peek(), SampleTrainingType)


def test_make_data_stream_source_from_multipart_formdata_file(
    sample_multipart_json,
    sample_multipart_csv,
    sample_multipart_json_with_content_header,
    tmp_path,
):
    """Test multipart streams. NB: We expect that multipart files will not have an extension"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType

    for file in (
        sample_multipart_csv,
        sample_multipart_json,
        sample_multipart_json_with_content_header,
    ):
        # prepare file with no extension
        no_extension_filename = os.path.join(str(tmp_path), "no_extension")
        shutil.copyfile(file, no_extension_filename)

        ds = stream_type(file=FileReference(filename=no_extension_filename))
        assert isinstance(ds, DataStreamSourceBase)

        data_stream = ds.to_data_stream()
        assert isinstance(data_stream, DataStream)

        validate_data_stream(data_stream, 2, SampleTrainingType)


#################
# List of files
#################


def test_make_data_stream_source_list_of_json_files(sample_json_file):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    # does NOT work with sample_json_collection as each file needs to have data in array
    ds = stream_type(
        list_of_files=stream_type.ListOfFileReferences(
            files=[sample_json_file, sample_json_file]
        )
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 4, SampleTrainingType)


def test_make_data_stream_source_list_of_csv_files(sample_csv_collection):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    sample_files = [
        os.path.join(sample_csv_collection, file)
        for file in os.listdir(sample_csv_collection)
    ]
    # send all files but last one
    ds = stream_type(list_of_files=stream_type.ListOfFileReferences(files=sample_files))

    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 6, SampleTrainingType)


def test_make_data_stream_source_list_of_jsonl_files(sample_jsonl_collection):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    sample_files = [
        os.path.join(sample_jsonl_collection, file)
        for file in os.listdir(sample_jsonl_collection)
    ]
    ds = stream_type(list_of_files=stream_type.ListOfFileReferences(files=sample_files))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 7, SampleTrainingType)


#################
# Directories
#################


def test_make_data_stream_source_jsondir(sample_json_collection):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(directory=Directory(dirname=sample_json_collection))
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 3, SampleTrainingType)


def test_make_data_stream_source_csvdir(sample_csv_collection):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        directory=Directory(dirname=sample_csv_collection, extension="csv")
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 6, SampleTrainingType)


def test_make_data_stream_source_jsonldir(sample_jsonl_collection):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        directory=Directory(dirname=sample_jsonl_collection, extension="jsonl")
    )
    assert isinstance(ds, DataStreamSourceBase)

    data_stream = ds.to_data_stream()
    assert isinstance(data_stream, DataStream)

    validate_data_stream(data_stream, 7, SampleTrainingType)


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


def test_make_data_stream_source_invalid_file_raises():
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(file=FileReference(filename="invalid_file"))
    with pytest.raises(CaikitRuntimeException):
        ds.to_data_stream()


def test_make_data_stream_source_invalid_dir_raises():
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(directory=Directory(dirname="invalid_dir"))
    with pytest.raises(CaikitRuntimeException):
        ds.to_data_stream()


def test_data_stream_source_single_oneof():
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    with pytest.raises(CaikitRuntimeException):
        stream_type(
            jsondata=stream_type.JsonData(data=[SampleTrainingType(1)]),
            directory=Directory(dirname="invalid_dir"),
        )


def test_make_data_stream_source_invalid_ext_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        fname = os.path.join(tempdir, f"sample.txt")
        with open(fname, "w") as handle:
            handle.write("something")
        stream_type = (
            caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
        )
        ds = stream_type(directory=Directory(dirname=tempdir, extension="txt"))
        with pytest.raises(CaikitRuntimeException) as e:
            ds.to_data_stream()
        assert "Extension not supported!" in e.value.message


def test_make_data_stream_source_no_files_w_ext_dir(sample_jsonl_collection):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    ds = stream_type(
        directory=Directory(dirname=sample_jsonl_collection, extension="csv")
    )
    with pytest.raises(CaikitRuntimeException) as e:
        ds.to_data_stream()
    assert "contains no source files with extension" in e.value.message


def test_make_data_stream_source_non_json_array_errors(tmp_path):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
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
    ds = stream_type(file=FileReference(filename=file_path))
    assert isinstance(ds, DataStreamSourceBase)

    with pytest.raises(ValueError, match="Non-array JSON object"):
        ds.to_data_stream()


def test_s3_not_implemented():
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


def test_datastream_sources_not_repeatedly_read(sample_json_file):
    """This test ensures that `to_data_stream` is only called once on a single instance of a
    DataStreamSource. This allows source plugin authors to control how data is cached for the
    life of the stream"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    data_stream = stream_type(file=FileReference(filename=sample_json_file))

    # Read the datastream as normal
    validate_data_stream(data_stream, 2, SampleTrainingType)

    # Delete the set source field so that .to_data_stream does not know how to find a source
    data_stream.file = None

    # Check that we can still read through the stream
    validate_data_stream(data_stream, 2, SampleTrainingType)
