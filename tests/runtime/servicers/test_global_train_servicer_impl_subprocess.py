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
"""Subprocess only global train servicer tests"""
# Third Party
import pytest

# Local
from caikit.core import MODEL_MANAGER
from caikit.runtime.service_factory import get_train_params, get_train_request
from sample_lib.data_model.sample import SampleTrainingType
from sample_lib.modules import SampleModule
from tests.conftest import random_test_id, set_use_subprocess
from tests.fixtures import Fixtures
import caikit.core


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

    # Enable sub-processing for test
    with set_use_subprocess(True):
        training_response = sample_train_servicer.Train(
            train_request, Fixtures.build_context("foo")
        )
        MODEL_MANAGER.get_model_future(training_response.training_id).wait()

        future_info = MODEL_MANAGER.get_model_future(
            training_response.training_id
        ).get_info()
        assert f"Training process died with OOM error!" in str(future_info.errors[0])
