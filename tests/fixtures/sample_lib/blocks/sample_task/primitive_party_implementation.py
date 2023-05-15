"""
A block meant to flex a bit of the protobufs primitive support
"""
# Standard
from typing import List

# Local
from ...data_model.sample import SampleOutputType, SampleTask
from caikit.core.module import ModuleSaver
import caikit.core


@caikit.core.block("00112233-0405-0607-0809-0a0b02dd0e0f", "SampleBlock", "0.0.1")
class SamplePrimitiveBlock(caikit.core.BlockBase):
    def __init__(self):
        super().__init__()

    @classmethod
    def load(cls, model_path, **kwargs):
        return cls()

    def run(
        self,
        bool_type: bool,
        int_type: int,
        float_type: float,
        str_type: str,
        bytes_type: bytes,
        list_type: List[str],
    ) -> SampleOutputType:
        """This takes in a bunch of primitive types to ensure that we can pass those through the runtime server correctly."""
        assert isinstance(bool_type, bool)
        assert isinstance(int_type, int)
        assert isinstance(float_type, float)
        assert isinstance(str_type, str)
        assert isinstance(bytes_type, bytes)
        return SampleOutputType(f"hello: primitives!")

    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            pass
