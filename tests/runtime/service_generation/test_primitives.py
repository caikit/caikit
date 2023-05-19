# Standard
from typing import Optional, Union

# Third Party
import pytest

# Local
from caikit.runtime.service_generation.primitives import (
    extract_data_model_type_from_union,
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
    ) == {"name": Union[str, int]}


def test_to_primitive_signature_dm():
    assert to_primitive_signature(
        signature={"name": SampleInputType},
        primitive_data_model_types=["sample_lib.data_model.SampleInputType"],
    ) == {"name": SampleInputType}


def test_to_primitive_signature_union_dm():
    assert to_primitive_signature(
        signature={"name": Union[SampleInputType, str]},
        primitive_data_model_types=["sample_lib.data_model.SampleInputType"],
    ) == {"name": Union[SampleInputType, str]}


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
