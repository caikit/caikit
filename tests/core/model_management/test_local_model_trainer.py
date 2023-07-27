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
Tests for LocalModelTrainer
"""
# Standard
from datetime import timedelta
import multiprocessing
import os
import tempfile
import threading

# Third Party
import pytest

# First Party
import aconfig

# Local
from caikit.core.data_model import DataStream, TrainingStatus
from caikit.core.model_management.local_model_trainer import LocalModelTrainer
from caikit.core.toolkit.destroyable_process import OOM_EXIT_CODE
from sample_lib.modules import SampleModule

## Helpers #####################################################################


def local_trainer(**kwargs) -> LocalModelTrainer:
    cfg = aconfig.Config(kwargs, override_env_vars=False)
    return LocalModelTrainer(cfg, "test-instance")


@pytest.fixture(params=[True, False])
def trainer_type_cfg(request):
    yield {"use_subprocess": request.param}


@pytest.fixture
def save_path():
    with tempfile.TemporaryDirectory() as workdir:
        yield os.path.join(workdir, "model_save_dir")


def get_event(cfg: dict):
    return cfg.get("use_subprocess") and multiprocessing.Event() or threading.Event()


## Tests #######################################################################


def test_train_and_get_info(trainer_type_cfg):
    """Test that running a training can fetch status correctly"""
    trainer = local_trainer(**trainer_type_cfg)

    # Launch the training and force it to wait
    # NOTE: Data stream passed by positional arg to ensure it is passed through
    #   correctly by position
    wait_event = get_event(trainer_type_cfg)
    model_future = trainer.train(
        SampleModule,
        DataStream.from_iterable([]),
        wait_event=wait_event,
    )
    assert model_future.get_info().status == TrainingStatus.RUNNING
    assert not model_future.get_info().status.is_terminal

    # Let the training proceed and wait for it to complete
    wait_event.set()
    model_future.wait()
    assert model_future.get_info().status == TrainingStatus.COMPLETED
    assert model_future.get_info().status.is_terminal

    # Re-fetch the future by ID
    fetched_future = trainer.get_model_future(model_future.id)
    assert fetched_future is model_future


def test_train_save_and_load(trainer_type_cfg, save_path):
    """Test that a trained model can be loaded"""
    trainer = local_trainer(**trainer_type_cfg)
    model_future = trainer.train(
        SampleModule,
        training_data=DataStream.from_iterable([]),
        save_path=save_path,
    )
    model_future.wait()

    # Make sure it can be loaded manually
    model = SampleModule.load(save_path)
    assert isinstance(model, SampleModule)

    # Make sure that it can be loaded via the future
    # NOTE: With a subprocess, this requires that save path is given so that the
    #   model can be re-loaded from disk
    model = model_future.load()
    assert isinstance(model, SampleModule)


def test_save_with_id(trainer_type_cfg, save_path):
    """Test that saving with the training id correctly injects the ID in the
    save path
    """
    trainer = local_trainer(**trainer_type_cfg)
    model_future = trainer.train(
        SampleModule,
        training_data=DataStream.from_iterable([]),
        save_path=save_path,
        save_with_id=True,
    )
    model_future.wait()
    assert model_future.save_path != save_path
    assert model_future.id in model_future.save_path
    assert os.path.exists(model_future.save_path)


def test_cancel_clean_termination(trainer_type_cfg):
    """Test that cancelling an in-progress training successfully destroys the
    training when the training is run in a way that can be shut down cleanly
    """
    trainer = local_trainer(**trainer_type_cfg)
    model_future = trainer.train(
        SampleModule,
        training_data=DataStream.from_iterable([]),
        sleep_time=1000,
    )
    assert model_future.get_info().status == TrainingStatus.RUNNING
    assert not model_future.get_info().status.is_terminal

    # Cancel the future
    model_future.cancel()
    assert model_future.get_info().status == TrainingStatus.CANCELED
    assert model_future.get_info().status.is_terminal
    model_future.wait()


def test_cancel_without_waiting(trainer_type_cfg):
    """Test that cancelling an in-progress training that uses a long sleep (and
    can't be easily destroyed in a thread) still reports CANCELED as the status
    before the training has fully terminated.
    """
    trainer = local_trainer(**trainer_type_cfg)
    model_future = trainer.train(
        SampleModule,
        training_data=DataStream.from_iterable([]),
        sleep_time=0.5,
        sleep_increment=0.5,
    )
    assert model_future.get_info().status == TrainingStatus.RUNNING
    assert not model_future.get_info().status.is_terminal

    # Cancel the future and make sure it reports canceled, even though the
    # function is still sleeping
    model_future.cancel()
    assert model_future.get_info().status == TrainingStatus.CANCELED
    assert model_future.get_info().status.is_terminal


def test_no_retention_time(trainer_type_cfg):
    """Test that constructing with no configured retention period keeps futures
    forever and doesn't cause errors
    """
    trainer = local_trainer(retention_duration=None, **trainer_type_cfg)
    model_future = trainer.train(SampleModule, DataStream.from_iterable([]))
    model_future.wait()
    retrieved_future = trainer.get_model_future(model_future.id)
    assert retrieved_future is model_future


def test_purge_retention_time(trainer_type_cfg):
    """Test that purging old models works as expected"""
    trainer = local_trainer(retention_duration="1d10s", **trainer_type_cfg)
    model_future = trainer.train(SampleModule, DataStream.from_iterable([]))
    model_future.wait()
    retrieved_future = trainer.get_model_future(model_future.id)
    assert retrieved_future is model_future
    model_future._completion_time = model_future._completion_time - timedelta(days=2)
    with pytest.raises(ValueError):
        trainer.get_model_future(model_future.id)


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
    trainer = local_trainer(retention_duration=test_params[0])
    assert trainer._retention_duration == test_params[1]


def test_get_into_return_error(trainer_type_cfg):
    """Test that failed training returns error properly"""
    trainer = local_trainer(**trainer_type_cfg)

    model_future = trainer.train(
        SampleModule,
        DataStream.from_iterable([]),
        batch_size=SampleModule.POISON_PILL_BATCH_SIZE,
    )
    # assert model_future.get_info().status == TrainingStatus.RUNNING
    # assert not model_future.get_info().status.is_terminal

    # Let the training proceed and wait for it to complete
    model_future.wait()
    assert model_future.get_info().status == TrainingStatus.ERRORED
    assert model_future.get_info().status.is_terminal
    assert isinstance(model_future.get_info().errors, list)
    assert isinstance(model_future.get_info().errors[0], ValueError)
    assert str(model_future.get_info().errors[0]) == "Batch size of 999 is not allowed!"
