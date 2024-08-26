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
"""
Tests for LocalJobPredictor
"""
# Standard
from datetime import timedelta
from pathlib import Path
import multiprocessing
import os
import tempfile
import threading

# Third Party
import pytest

# First Party
import aconfig

# Local
from caikit.config import get_config
from caikit.core import ModuleBase
from caikit.core.data_model import DataStream, PredictionJobStatus
from caikit.core.exceptions.caikit_core_exception import CaikitCoreException
from caikit.core.model_management.local_job_predictor import LocalJobPredictor
from sample_lib.data_model import SampleInputType, SampleOutputType
from sample_lib.modules import SampleModule

## Helpers #####################################################################


def local_predictor(**kwargs) -> LocalJobPredictor:
    cfg = aconfig.Config(kwargs, override_env_vars=False)
    return LocalJobPredictor(cfg, "test-instance")


@pytest.fixture
def save_path():
    with tempfile.TemporaryDirectory() as workdir:
        yield os.path.join(workdir, "model_save_dir")


@pytest.fixture()
def predictor_type_cfg(save_path):
    yield {"result_dir": save_path}


class WaitPredict:
    """Dummy module that will block prediction on an event"""

    def run(self, wait_event: threading.Event) -> SampleOutputType:
        wait_event.wait()
        return SampleOutputType(greeting="hello world")


## Tests #######################################################################


def test_predict_and_get_info(predictor_type_cfg):
    """Test that running a prediction can fetch status correctly"""
    predictor = local_predictor(**predictor_type_cfg)

    # Launch the training and force it to wait
    # NOTE: Data stream passed by positional arg to ensure it is passed through
    #   correctly by position
    wait_event = threading.Event()
    job_future = predictor.predict(
        WaitPredict(),
        "run",
        wait_event=wait_event,
    )
    assert job_future.get_info().status == PredictionJobStatus.RUNNING
    assert not job_future.get_info().status.is_terminal

    # Let the training proceed and wait for it to complete
    wait_event.set()
    job_future.wait()
    assert job_future.get_info().status == PredictionJobStatus.COMPLETED
    assert job_future.get_info().status.is_terminal

    # Re-fetch the future by ID
    fetched_future = predictor.get_prediction_future(job_future.id)
    assert fetched_future is job_future


def test_save_with_id(predictor_type_cfg, save_path):
    """Test that saving with the result id correctly injects the ID in the
    save path
    """
    predictor = local_predictor(**predictor_type_cfg)
    job_future = predictor.predict(
        SampleModule(),
        "run",
        sample_input=SampleInputType(),
    )
    job_future.wait()
    assert job_future.save_path != save_path
    assert job_future.id in job_future.save_path
    assert os.path.exists(job_future.save_path)


def test_cancel_clean_termination(predictor_type_cfg):
    """Test that cancelling an in-progress prediction successfully destroys the
    prediction
    """
    predictor = local_predictor(**predictor_type_cfg)
    job_future = predictor.predict(
        SampleModule(),
        "run",
        sample_input=SampleInputType(),
        sleep_time=1000,
    )
    assert job_future.get_info().status == PredictionJobStatus.RUNNING
    assert not job_future.get_info().status.is_terminal

    # Cancel the future
    job_future.cancel()
    assert job_future.get_info().status == PredictionJobStatus.CANCELED
    assert job_future.get_info().status.is_terminal
    job_future.wait()


def test_cancel_without_waiting(predictor_type_cfg):
    """Test that cancelling an in-progress prediction that uses a long sleep (and
    can't be easily destroyed in a thread) still reports CANCELED as the status
    before the prediction has fully terminated.
    """
    predictor = local_predictor(**predictor_type_cfg)
    job_future = predictor.predict(
        SampleModule(),
        "run",
        sample_input=SampleInputType(),
        sleep_time=1000,
        sleep_increment=0.5,
    )
    assert job_future.get_info().status == PredictionJobStatus.RUNNING
    assert not job_future.get_info().status.is_terminal

    # Cancel the future and make sure it reports canceled, even though the
    # function is still sleeping
    job_future.cancel()
    assert job_future.get_info().status == PredictionJobStatus.CANCELED
    assert job_future.get_info().status.is_terminal
    job_future.wait()


def test_no_retention_time(predictor_type_cfg):
    """Test that constructing with no configured retention period keeps futures
    forever and doesn't cause errors
    """
    predictor = local_predictor(retention_duration=None, **predictor_type_cfg)
    job_future = predictor.predict(
        SampleModule(),
        "run",
        sample_input=SampleInputType(),
    )
    job_future.wait()
    retrieved_future = predictor.get_prediction_future(job_future.id)
    assert retrieved_future is job_future


def test_purge_retention_time(predictor_type_cfg):
    """Test that purging old models works as expected"""
    predictor = local_predictor(retention_duration="1d10s", **predictor_type_cfg)
    job_future = predictor.predict(
        SampleModule(),
        "run",
        sample_input=SampleInputType(),
    )
    job_future.wait()
    retrieved_future = predictor.get_prediction_future(job_future.id)
    assert retrieved_future is job_future
    job_future._completion_time = job_future._completion_time - timedelta(days=2)
    with pytest.raises(CaikitCoreException):
        predictor.get_prediction_future(job_future.id)
    assert not Path(retrieved_future.save_path).exists()
    assert not Path(retrieved_future.save_path).parent.exists()


@pytest.mark.parametrize(
    "test_params",
    [
        ("1d", timedelta(days=1)),
        ("0.01s", timedelta(seconds=0.01)),
        ("1d12h3m0.2s", timedelta(days=1, hours=12, minutes=3, seconds=0.2)),
    ],
)
def test_retention_duration_parse(test_params):
    """Make sure the regex for the duration can parse all expected durations"""
    trainer = local_predictor(retention_duration=test_params[0])
    assert trainer._retention_duration == test_params[1]


def test_get_into_return_error(predictor_type_cfg):
    """Test that failed prediction returns error properly"""
    predictor = local_predictor(**predictor_type_cfg)

    job_future = predictor.predict(
        SampleModule(),
        "run",
        sample_input=SampleInputType("hello"),
        throw=True,
        error="Error",
    )

    # Let the training proceed and wait for it to complete
    job_future.wait()
    assert job_future.get_info().status == PredictionJobStatus.ERRORED
    assert job_future.get_info().status.is_terminal
    assert isinstance(job_future.get_info().errors, list)
    assert isinstance(job_future.get_info().errors[0], RuntimeError)
    assert str(job_future.get_info().errors[0]) == "Error"


def test_retry_duplicate_external_id():
    """Test that a prediction can be retried safely reusing an external ID"""
    predictor = local_predictor()
    predictor_id = "my-prediction"

    # First try should fail
    with pytest.raises(CaikitCoreException):
        job_future = predictor.predict(
            SampleModule(),
            "run",
            external_prediction_id=predictor_id,
            sample_input=SampleInputType("hello"),
            throw=True,
            error="Error",
        )
        job_future.result()

    # Second time should succeed
    job_future = predictor.predict(
        SampleModule(),
        "run",
        external_prediction_id=predictor_id,
        sample_input=SampleInputType("hello"),
    )
    assert job_future.result()


def test_duplicate_external_id_cannot_restart_while_running():
    """Make sure that if a prediction is actively running, it cannot be replaced
    by a rerun
    """
    predictor = local_predictor()
    predictor_id = "my-prediction"
    wait_event = threading.Event()

    job_future = predictor.predict(
        WaitPredict(), "run", external_prediction_id=predictor_id, wait_event=wait_event
    )
    # Test that rerunning existing prediction fails
    with pytest.raises(ValueError, match="Cannot restart prediction.*"):
        job_future = predictor.predict(
            model_instance=WaitPredict(),
            prediction_func_name="run",
            external_prediction_id=predictor_id,
            wait_event=wait_event,
        )

    assert predictor.get_prediction_future(predictor_id) is job_future
    wait_event.set()
    assert job_future.result()
