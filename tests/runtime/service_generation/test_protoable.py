# Standard
from typing import Dict, List, Optional, Union
import json

# Third Party
import pytest

# First Party
from py_to_proto.dataclass_to_proto import Annotated

# Local
from caikit.core.data_model.base import DataBase
from caikit.runtime.service_generation.protoable import (
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


def test_to_protoable_signature_list_w_primitive():
    union_list_signature = to_protoable_signature(
        signature={"union_list": Union[List[str], int]},
    )
    assert union_list_signature == {
        "union_list": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_str_sequence",
            ],
            int,
        ]
    }


def test_to_protoable_signature_multiple_same_type_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int]],
            "union_list_another_arg": Union[List[str], List[int]],
        },
    )
    assert_sig = {
        "union_list_arg": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_arg_str_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.IntSequence,
                "union_list_arg_int_sequence",
            ],
        ],
        "union_list_another_arg": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_another_arg_str_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.IntSequence,
                "union_list_another_arg_int_sequence",
            ],
        ],
    }
    assert all((union_list_signature.get(k) == v for k, v in assert_sig.items()))


def test_to_protoable_signature_all_types_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int], List[float], List[bool]],
        },
    )
    assert union_list_signature == {
        "union_list_arg": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_arg_str_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.IntSequence,
                "union_list_arg_int_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.FloatSequence,
                "union_list_arg_float_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.BoolSequence,
                "union_list_arg_bool_sequence",
            ],
        ]
    }


def test_to_protoable_signature_throws_unknown_type():
    # Third Party
    import numpy as np

    with pytest.raises(AttributeError):
        to_protoable_signature(
            signature={
                "union_list_arg": Union[List[np.int32], List[int]],
            },
        )


def test_to_protoable_signature_empty_list_arg():
    """Empty list arg is ignored by is_protoable_type"""
    assert to_protoable_signature(
        signature={
            "union_list_arg": Union[List, int],
        },
    ) == {"union_list_arg": int}


def test_to_protoable_signature_multiple_diff_type_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int]],
            "union_list_another_arg": Union[List[str], List[bool]],
        },
    )
    assert_sig = {
        "union_list_arg": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_arg_str_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.IntSequence,
                "union_list_arg_int_sequence",
            ],
        ],
        "union_list_another_arg": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_another_arg_str_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.BoolSequence,
                "union_list_another_arg_bool_sequence",
            ],
        ],
    }
    assert all((union_list_signature.get(k) == v for k, v in assert_sig.items()))


def test_to_protoable_signature_union_list():
    union_list_signature = to_protoable_signature(
        signature={
            "union_list_arg": Union[List[str], List[int]],
        },
    )
    assert union_list_signature == {
        "union_list_arg": Union[
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.StrSequence,
                "union_list_arg_str_sequence",
            ],
            Annotated[
                caikit.interfaces.common.data_model.primitive_sequences.IntSequence,
                "union_list_arg_int_sequence",
            ],
        ]
    }

    # str sequence
    str_seq = caikit.interfaces.common.data_model.primitive_sequences.StrSequence(
        values=["one", "two"]
    )

    ## proto test
    str_seq.from_proto(str_seq.to_proto()) == str_seq
    ## json test
    union_list_str_json_repr = {"values": ["one", "two"]}
    assert str_seq.to_json() == json.dumps(union_list_str_json_repr)
    assert str_seq.from_json(union_list_str_json_repr) == str_seq

    # int Intuence
    int_seq = caikit.interfaces.common.data_model.primitive_sequences.IntSequence(
        values=[1, 2]
    )

    ## proto test
    int_seq.from_proto(int_seq.to_proto()) == int_seq
    ## json test
    union_list_int_json_repr = {"values": [1, 2]}
    assert int_seq.to_json() == json.dumps(union_list_int_json_repr)
    assert int_seq.from_json(union_list_int_json_repr) == int_seq


################################################################
# get_protoable_return_type
################################################################


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
