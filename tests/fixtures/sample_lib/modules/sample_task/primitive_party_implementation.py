"""
A module meant to flex a bit of the protobufs primitive support
"""
# Standard
from dataclasses import field
from typing import Dict, List, Union

# Local
from ...data_model.sample import SampleInputType, SampleOutputType, SampleTask
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit.core


@caikit.core.module(
    "00112233-0405-0607-0809-0a0b02dd0e0f", "SampleModule", "0.0.1", SampleTask
)
class SamplePrimitiveModule(caikit.core.ModuleBase):
    def __init__(self, training_params_json_dict=None, training_params_dict={}):
        super().__init__()
        self.training_params_json_dict = training_params_json_dict
        self.training_params_dict = training_params_dict

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls(
            config["train"]["training_params_json_dict"],
            config["train"]["training_params_dict"],
        )

    def run(
        self,
        sample_input: SampleInputType,
        bool_type: bool = True,
        int_type: int = 42,
        float_type: float = 34.0,
        str_type: str = "moose",
        bytes_type: bytes = b"",
        list_type: List[str] = None,
    ) -> SampleOutputType:
        """This takes in a bunch of primitive types to ensure that we can pass those through the runtime server correctly."""
        assert isinstance(bool_type, bool)
        assert isinstance(int_type, int)
        assert isinstance(float_type, float)
        assert isinstance(str_type, str)
        assert isinstance(bytes_type, bytes)
        return SampleOutputType(
            f"hello: primitives! {self.training_params_json_dict.get('foo').get('bar')} {self.training_params_dict.get('layer_sizes')}"
        )

    def save(self, model_path):
        module_saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            config_options = {
                "train": {
                    "training_params_json_dict": self.training_params_json_dict,
                    "training_params_dict": self.training_params_dict,
                },
            }

            module_saver.update_config(config_options)

    @classmethod
    def train(
        cls,
        sample_input: SampleInputType,
        simple_list: List[str],
        union_list: Union[List[str], List[int]],
        union_list2: Union[List[str], List[int], int],
        union_list3: Union[List[str], List[bool]],
        union_list4: Union[List[str], int],
        training_params_json_dict_list: List[JsonDict],
        training_params_json_dict: JsonDict = None,
        training_params_dict: Dict[str, int] = field(default_factory=dict),
        training_params_dict_int: Dict[int, float] = field(default_factory=dict),
    ) -> "SamplePrimitiveModule":
        """Sample training method that produces a trained model"""
        assert type(sample_input) == SampleInputType
        assert isinstance(simple_list, List)
        assert isinstance(union_list.values, List)
        assert isinstance(union_list2.values, List)
        assert isinstance(union_list3.values, List)
        assert isinstance(union_list4, int)
        assert isinstance(training_params_json_dict_list, List)
        assert isinstance(training_params_json_dict, Dict)
        assert isinstance(training_params_dict, Dict)
        assert isinstance(training_params_dict_int, Dict)
        assert training_params_json_dict is not None
        assert len(training_params_json_dict_list) > 0
        assert len(union_list.values) > 0
        assert len(union_list2.values) > 0
        assert len(union_list3.values) > 0
        assert training_params_dict is not None
        assert training_params_dict_int is not None
        return cls(
            training_params_json_dict=training_params_json_dict,
            training_params_dict=training_params_dict,
        )
