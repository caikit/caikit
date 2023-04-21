# Standard
from typing import Optional, Union

# Third Party
import pytest

# Local
from caikit.runtime.service_generation.primitives import (
    extract_data_model_type_from_union,
    extract_primitive_type_from_union,
    to_primitive_signature,
)
from sample_lib.data_model import SampleInputType, SampleOutputType


def test_to_primitive_signature_raw():
    assert to_primitive_signature(
        signature={"name": str},
        primitive_data_model_types=[],
    ) == {"name": str}


def test_to_primitive_signature_union_raw():
    assert to_primitive_signature(
        signature={"name": Union[str, int]},
        primitive_data_model_types=[],
    ) == {"name": str}


def test_to_primitive_signature_dm():
    assert to_primitive_signature(
        signature={"name": SampleInputType},
        primitive_data_model_types=["sample_lib.data_model.SampleInputType"],
    ) == {"name": SampleInputType}


def test_to_primitive_signature_union_dm():
    assert to_primitive_signature(
        signature={"name": Union[SampleInputType, str]},
        primitive_data_model_types=["sample_lib.data_model.SampleInputType"],
    ) == {"name": SampleInputType}


def test_to_primitive_signature_no_primitive():
    assert (
        to_primitive_signature(
            signature={"name": SampleInputType},
            primitive_data_model_types=[],
        )
        == {}
    )


def test_to_output_dm_type_with_None():
    assert extract_data_model_type_from_union(None) == None


def test_to_output_dm_type_with_raw_primitive():
    assert extract_data_model_type_from_union(str) == str


def test_to_output_dm_type_with_dm():
    assert extract_data_model_type_from_union(SampleOutputType) == SampleOutputType


def test_to_output_dm_type_with_union_dm():
    assert (
        extract_data_model_type_from_union(Union[SampleOutputType, str])
        == SampleOutputType
    )


def test_to_output_dm_type_with_union_optional_dm():
    assert (
        extract_data_model_type_from_union(Union[Optional[SampleOutputType], str])
        == SampleOutputType
    )


def test_extract_primitive_dm_type_from_union():
    """If we provide a list of primitive_data_model_types,
    then fetch that from the Union"""
    assert (
        extract_primitive_type_from_union(
            primitive_data_model_types=["sample_lib.data_model.SampleInputType"],
            arg_type=Union[SampleInputType, str],
        )
        == SampleInputType
    )


def test_extract_primitive_raw_type_from_union():
    """If we don't provide a primitive_data_model_types, then
    fetch the raw primitive from union"""
    assert (
        extract_primitive_type_from_union(
            primitive_data_model_types=[],
            arg_type=Union[SampleInputType, str],
        )
        == str
    )


def test_extract_primitive_multiple_raw_type_union():
    """If multiple raw primitives exist, return the first one"""
    assert (
        extract_primitive_type_from_union(
            primitive_data_model_types=[], arg_type=Union[str, int]
        )
        == str
    )
    assert (
        extract_primitive_type_from_union(
            primitive_data_model_types=[], arg_type=Union[int, str]
        )
        == int
    )


def test_extract_primitive_raw_type_no_union():
    """If we provide a primitive without a union, we still get it back"""
    assert (
        extract_primitive_type_from_union(primitive_data_model_types=[], arg_type=str)
        == str
    )


def test_extract_primitive_dm_type_no_union():
    """If we provide a primitive dm without a union, we still get it back"""
    assert (
        extract_primitive_type_from_union(
            primitive_data_model_types=["sample_lib.data_model.SampleInputType"],
            arg_type=SampleInputType,
        )
        == SampleInputType
    )


def test_extract_primitive_type_no_primitive():
    """If no primitives exist, nothing gets returned"""
    assert (
        extract_primitive_type_from_union(
            primitive_data_model_types=[], arg_type=SampleInputType
        )
        == None
    )
