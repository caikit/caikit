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
"""Subprocess only model train servicer tests"""

# Standard
import json
import uuid

# Third Party
import grpc
import pytest

# Local
from .test_model_train_servicer_impl import sample_model_train_servicer
from caikit.runtime.protobufs import process_pb2
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.conftest import set_use_subprocess
from tests.fixtures import Fixtures
import sample_lib


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
