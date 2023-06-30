# Standard
from typing import Dict, List, Optional, Union
import json

# Third Party
import pytest

# Local
from caikit.core.data_model.base import DataBase
from caikit.runtime.service_generation.protoable import (
    _make_union_list_source_type_name,
    get_protoable_return_type,
    to_protoable_signature,
)
from sample_lib.data_model import SampleInputType, SampleOutputType
import caikit


# Class that does not have to/from_proto
class NonProtoable:
    pass


################################################################
# to_protoable_signature
################################################################


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
    assert get_protoable_return_type(None) == None


def test_to_output_dm_type_with_raw_primitive():
    assert get_protoable_return_type(str) == str


def test_to_output_dm_type_with_dm():
    assert get_protoable_return_type(SampleOutputType) == SampleOutputType


def test_to_output_dm_type_with_union_dm():
    assert get_protoable_return_type(Union[SampleOutputType, str]) == SampleOutputType


def test_to_output_dm_type_with_union_optional_dm():
    assert (
        get_protoable_return_type(Union[Optional[SampleOutputType], str])
        == SampleOutputType
    )


def test_to_protoable_signature_dict():
    assert to_protoable_signature(
        signature={"name": Dict[str, int]},
    ) == {"name": Dict[str, int]}


def test_to_protoable_signature_dict_int_keys():
    assert to_protoable_signature(
        signature={"name": Dict[int, float]},
    ) == {"name": Dict[int, float]}


def test_to_protoable_signature_unsupported_dict():
    assert (
        to_protoable_signature(
            signature={"name": Dict[str, NonProtoable]},
        )
        == {}
    )


def test_to_protoable_signature_dict_incomplete_type_hint():
    assert (
        to_protoable_signature(
            signature={"name": dict},
        )
        == {}
    )


def test_to_protoable_signature_optional_list():
    union_list_signature = to_protoable_signature(
        signature={"optional_param": Optional[List[str]]},
    )
    assert union_list_signature == {"optional_param": List[str]}


def test_to_protoable_signature_multiple_same_type_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int]],
            "union_list_another_arg": Union[List[str], List[int]],
        },
    )
    assert_sig = {
        "union_list_arg": caikit.interfaces.common.data_model.UnionListStrIntSource,
        "union_list_another_arg": caikit.interfaces.common.data_model.UnionListStrIntSource,
    }
    assert all((union_list_signature.get(k) == v for k, v in assert_sig.items()))


def test_to_protoable_signature_multiple_diff_type_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int]],
            "union_list_another_arg": Union[List[str], List[bool]],
        },
    )
    assert_sig = {
        "union_list_arg": caikit.interfaces.common.data_model.UnionListStrIntSource,
        "union_list_another_arg": caikit.interfaces.common.data_model.UnionListStrBoolSource,
    }
    assert all((union_list_signature.get(k) == v for k, v in assert_sig.items()))


def test_to_protoable_signature_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int]],
        },
    )
    assert union_list_signature == {
        "union_list_arg": caikit.interfaces.common.data_model.UnionListStrIntSource,
    }
    union_list_dm = union_list_signature["union_list_arg"]
    assert issubclass(union_list_dm, DataBase)

    # str sequence
    union_list_str_instance = union_list_dm(
        union_list=union_list_dm.StrSequence(values=["one", "two"])
    )
    ## proto test
    union_list_dm.from_proto(
        union_list_str_instance.to_proto()
    ) == union_list_str_instance
    ## json test
    union_list_str_json_repr = {"strsequence": {"values": ["one", "two"]}}
    assert union_list_str_instance.to_json() == json.dumps(union_list_str_json_repr)
    assert union_list_dm.from_json(union_list_str_json_repr) == union_list_str_instance

    # int sequence
    union_list_int_instance = union_list_dm(
        union_list=union_list_dm.IntSequence(values=[1, 2])
    )
    ## proto test
    union_list_dm.from_proto(
        union_list_int_instance.to_proto()
    ) == union_list_int_instance
    union_list_int_json_repr = {"intsequence": {"values": [1, 2]}}
    assert union_list_int_instance.to_json() == json.dumps(union_list_int_json_repr)
    assert union_list_dm.from_json(union_list_int_json_repr) == union_list_int_instance


################################################################
# _make_union_list_source_type_name
################################################################


def test_make_union_list_source_type_name():
    assert (
        _make_union_list_source_type_name([List[str], List[int]])
        == "UnionListStrIntSource"
    )
    assert (
        _make_union_list_source_type_name([List[str], List[bool]])
        == "UnionListStrBoolSource"
    )
