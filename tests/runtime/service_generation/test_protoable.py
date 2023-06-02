# Standard
from typing import List, Optional, Union

# Third Party
import pytest

# Local
from caikit.runtime.service_generation.protoable import (
    extract_data_model_type_from_union,
    to_protoable_signature,
)
from sample_lib.data_model import SampleInputType, SampleOutputType


# Class that does not have to/from_proto
class NonProtoable:
    pass


def test_to_primitive_signature_raw():
    assert to_protoable_signature(
        signature={"name": str},
    ) == {"name": str}


def test_to_protoable_signature_union_raw():
    assert to_protoable_signature(
        signature={"name": Union[str, int]},
    ) == {"name": Union[str, int]}


def test_to_protoable_signature_lists():
    assert to_protoable_signature(
        signature={"good_list": List[str], "bad_list": List, "lowercase_l_list": list},
    ) == {"good_list": List[str]}


def test_to_protoable_signature_dm():
    assert to_protoable_signature(
        signature={"name": SampleInputType},
    ) == {"name": SampleInputType}


def test_to_protoable_signature_union_dm():
    assert to_protoable_signature(
        signature={"name": Union[SampleInputType, str]},
    ) == {"name": Union[SampleInputType, str]}


def test_to_protoable_signature_unsupported_type_in_union():
    assert to_protoable_signature(
        signature={"name": Union[NonProtoable, str]},
    ) == {"name": str}


def test_to_protoable_signature_no_dm_types_in_union():
    class AnotherNonProtoable:
        pass

    assert to_protoable_signature(
        signature={"name": Union[NonProtoable, AnotherNonProtoable, str]},
    ) == {"name": str}


def test_to_protoable_signature_multiple_types_in_union():
    """We have the first arg as a supported DM arg, and the last as
    a supported primitive arg. We return the first supported DM arg"""
    assert to_protoable_signature(
        signature={"name": Union[SampleInputType, NonProtoable, str]},
    ) == {"name": SampleInputType}


def test_to_protoable_signature_multiple_no_dm_types_in_union():
    """We have the first arg that's not supported, and 2 primitive args. We
    return the first primitive arg"""
    assert to_protoable_signature(
        signature={"name": Union[NonProtoable, str, int]},
    ) == {"name": str}


def test_to_protoable_signature_no_protoable_types():
    assert (
        to_protoable_signature(
            signature={"name": NonProtoable},
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
