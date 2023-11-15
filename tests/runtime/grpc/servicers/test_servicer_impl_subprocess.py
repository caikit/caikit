# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Subprocess only tests - includes global train servicer and model train 
servicer tests"""

# Standard
import json
import uuid

# Third Party
import grpc
import pytest

# Local
from .test_model_train_servicer_impl import sample_model_train_servicer
from caikit.core import MODEL_MANAGER
from caikit.runtime.protobufs import process_pb2
from caikit.runtime.service_factory import get_train_params, get_train_request
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from sample_lib.data_model.sample import SampleTrainingType
from sample_lib.modules import SampleModule
from tests.conftest import random_test_id, set_use_subprocess
from tests.fixtures import Fixtures
import caikit.core
import sample_lib


@pytest.mark.parametrize("oom_exit_code", [137, 9])
def test_global_train_returns_exit_code_with_oom(
    sample_train_servicer, reset_model_manager, oom_exit_code
):
    """Test that if module goes into OOM we are able to surface error code"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[SampleTrainingType(1)])
    )
    train_class = get_train_request(SampleModule)
    train_request_params_class = get_train_params(SampleModule)
    train_request = train_class(
        model_name=random_test_id(),
        parameters=train_request_params_class(
            batch_size=42,
            training_data=training_data,
            oom_exit=True,
            oom_exit_code=oom_exit_code,
        ),
    ).to_proto()

    with set_use_subprocess(True):
        training_response = sample_train_servicer.Train(
            train_request, Fixtures.build_context("foo")
        )
        MODEL_MANAGER.get_model_future(training_response.training_id).wait()

        future_info = MODEL_MANAGER.get_model_future(
            training_response.training_id
        ).get_info()
        assert f"Training process died with OOM error!" in str(future_info.errors[0])


@pytest.mark.parametrize("oom_exit_code", [137, 9])
def test_model_train_memory_error_raises(
    sample_model_train_servicer, reset_model_manager, oom_exit_code
):
    """Test that if there is memory error we are able to raise it"""
    with pytest.raises(CaikitRuntimeException) as e:
        context = Fixtures.build_context("foo")

        training_id = str(uuid.uuid4())
        model_train_request = process_pb2.ProcessRequest(
            trainingID=training_id,
            request_dict={
                "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
                "training_params": json.dumps(
                    {
                        "model_name": "abc",
                        "parameters": {
                            "training_data": {
                                "jsondata": {
                                    "data": [
                                        sample_lib.data_model.SampleTrainingType(
                                            number=1
                                        ).to_dict()
                                    ]
                                },
                            },
                            "oom_exit": True,
                            "oom_exit_code": oom_exit_code,
                        },
                    }
                ),
            },
            training_input_dir="training_input_dir",
        )
        with set_use_subprocess(True):
            sample_model_train_servicer.Run(model_train_request, context)

    assert "OOM error during training" in str(e.value)
    assert e.value.status_code == grpc.StatusCode.INTERNAL
