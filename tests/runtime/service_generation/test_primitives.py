# Standard
from typing import Optional, Union

# Third Party
import pytest

# Local
from caikit.runtime.service_generation.primitives import to_output_dm_type
from sample_lib.data_model import SampleOutputType


def test_to_output_dm_type_with_dm():
    assert to_output_dm_type(SampleOutputType) == SampleOutputType


def test_to_output_dm_type_with_union_dm():
    assert to_output_dm_type(Union[SampleOutputType, str]) == SampleOutputType


def test_to_output_dm_type_with_union_optional_dm():
    assert to_output_dm_type(Union[Optional[SampleOutputType], str]) == SampleOutputType


def test_to_output_dm_type_raises_primitive():
    with pytest.raises(RuntimeError):
        to_output_dm_type(str)
