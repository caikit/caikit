"""
Unit tests for the train script entrypoint
"""

# Standard
from contextlib import contextmanager
from unittest import mock
import copy
import json
import os
import sys
import tempfile

# Third Party
import pytest

# Local
from caikit.core.registries import module_registry
from caikit.runtime import train
from caikit.runtime.train import main
from sample_lib.modules import SampleModule
from tests.conftest import reset_module_registry, temp_config

## Helpers #####################################################################


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as workdir:
        yield workdir


@contextmanager
def sys_argv(*args):
    with mock.patch.object(sys, "argv", ["train.py"] + list(args)):
        yield


SAMPLE_MODULE = f"{SampleModule.__module__}.{SampleModule.__name__}"
SAMPLE_TRAIN_KWARGS = {
    "training_data": {
        "jsondata": {
            "data": [
                {"number": 1, "label": "foo"},
                {"number": 2, "label": "bar"},
            ],
        },
    },
}

## Tests #######################################################################


def test_train_sample_module(workdir):
    """Test performing a simple training using the script"""
    model_name = "my-model"
    with sys_argv(
        "--module",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
        "--save-path",
        workdir,
        "--training-kwargs",
        json.dumps(SAMPLE_TRAIN_KWARGS),
    ):
        assert main() == 0
        model_dir = os.path.join(workdir, model_name)
        assert os.path.isdir(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_train_from_file(workdir):
    """Test training using a file with the request kwargs"""
    model_name = "my-model"
    train_kwargs_file = os.path.join(workdir, "train.json")
    with open(train_kwargs_file, "w") as handle:
        handle.write(json.dumps(SAMPLE_TRAIN_KWARGS))
    with sys_argv(
        "--module",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
        "--save-path",
        workdir,
        "--training-kwargs",
        train_kwargs_file,
    ):
        assert main() == 0
        model_dir = os.path.join(workdir, model_name)
        assert os.path.isdir(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_train_module_uid(workdir):
    """Test referencing the module by its UID"""
    model_name = "my-model"
    with sys_argv(
        "--module",
        SampleModule.MODULE_ID,
        "--model-name",
        model_name,
        "--save-path",
        workdir,
        "--training-kwargs",
        json.dumps(SAMPLE_TRAIN_KWARGS),
    ):
        assert main() == 0
        model_dir = os.path.join(workdir, model_name)
        assert os.path.isdir(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_train_save_with_id(workdir):
    """Test saving with the training ID"""
    model_name = "my-model"
    with sys_argv(
        "--module",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
        "--save-path",
        workdir,
        "--training-kwargs",
        json.dumps(SAMPLE_TRAIN_KWARGS),
        "--save-with-id",
    ):
        assert main() == 0
        flat_model_dir = os.path.join(workdir, model_name)
        assert not os.path.isdir(flat_model_dir)
        dirs = list(
            filter(
                lambda fname: os.path.isdir(fname),
                [os.path.join(workdir, fname) for fname in os.listdir(workdir)],
            )
        )
        assert len(dirs) == 1
        assert os.path.isfile(os.path.join(dirs[0], model_name, "config.yml"))


def test_train_non_default_trainer(workdir):
    """Test that a non-default trainer can be used"""
    model_name = "my-model"
    other_trainer = "other"
    with temp_config(
        {
            "model_management": {
                "trainers": {
                    "default": {
                        "type": "INVALID",
                    },
                    other_trainer: {
                        "type": "LOCAL",
                        "config": {
                            "use_subprocess": False,
                        },
                    },
                }
            }
        },
        "merge",
    ):
        with sys_argv(
            "--module",
            SAMPLE_MODULE,
            "--model-name",
            model_name,
            "--save-path",
            workdir,
            "--training-kwargs",
            json.dumps(SAMPLE_TRAIN_KWARGS),
            "--trainer",
            other_trainer,
        ):
            assert main() == 0
            model_dir = os.path.join(workdir, model_name)
            assert os.path.isdir(model_dir)
            assert os.path.isfile(os.path.join(model_dir, "config.yml"))


def test_train_import_library(workdir, reset_module_registry):
    """Test that the --library arg can be used to import a library for a module"""
    model_name = "my-model"
    with mock.patch("importlib.import_module") as import_module_mock:
        with sys_argv(
            "--module",
            SampleModule.MODULE_ID,
            "--model-name",
            model_name,
            "--save-path",
            workdir,
            "--training-kwargs",
            json.dumps(SAMPLE_TRAIN_KWARGS),
            "--library",
            "sample_lib",
        ):
            assert main() == 0
            model_dir = os.path.join(workdir, model_name)
            assert os.path.isdir(model_dir)
            assert os.path.isfile(os.path.join(model_dir, "config.yml"))
            import_module_mock.assert_called()
            assert [call.args for call in import_module_mock.call_args_list] == [
                ("sample_lib",),
                (SampleModule.MODULE_ID,),
            ]


def test_invalid_json():
    """Make sure that an exception is raised for invalid json"""
    model_name = "my-model"
    with sys_argv(
        "--module",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
        "--training-kwargs",
        "{invalid json",
    ):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == train.USER_ERROR_EXIT_CODE


def test_failed_training():
    """Make sure that a non-zero exit code is returned if training fails"""
    model_name = "my-model"
    training_kwargs = copy.deepcopy(SAMPLE_TRAIN_KWARGS)
    training_kwargs["batch_size"] = SampleModule.POISON_PILL_BATCH_SIZE
    with sys_argv(
        "--module",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
        "--training-kwargs",
        json.dumps(training_kwargs),
    ):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == train.INTERNAL_ERROR_EXIT_CODE


def test_bad_module():
    """Make sure that a non-zero exit code is returned if an invalid module is provided"""
    model_name = "my-model"
    training_kwargs = copy.deepcopy(SAMPLE_TRAIN_KWARGS)
    with sys_argv(
        "--module",
        "this.is.a.bad.module",
        "--model-name",
        model_name,
        "--training-kwargs",
        json.dumps(training_kwargs),
    ):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == train.USER_ERROR_EXIT_CODE


def test_no_module_provided():
    """Make sure that a non-zero exit code is returned if an invalid module is provided"""
    model_name = "my-model"
    training_kwargs = copy.deepcopy(SAMPLE_TRAIN_KWARGS)
    with sys_argv(
        "--model-name",
        model_name,
        "--training-kwargs",
        json.dumps(training_kwargs),
    ):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 1


def test_blank_kwargs():
    """Make sure that a non-zero exit code is returned if kwargs are blank"""
    model_name = "my-model"
    with sys_argv(
        "--model-name",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
    ):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == train.USER_ERROR_EXIT_CODE


def test_empty_module_name():
    """Test handling of empty module parameter"""
    model_name = "my-model"
    with sys_argv(
        "--module",
        "",
        "--model-name",
        model_name,
        "--training-kwargs",
        json.dumps(SAMPLE_TRAIN_KWARGS),
    ):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == train.USER_ERROR_EXIT_CODE


def test_non_existent_save_path():
    """Test with a non-existent save path"""
    # We cannot verify save path ahead of time, so if it is unable
    # to be written to, the training will fail with a system error
    model_name = "my-model"
    non_existent_path = "/path/that/does/not/exist"
    with sys_argv(
        "--module",
        SAMPLE_MODULE,
        "--model-name",
        model_name,
        "--save-path",
        non_existent_path,
        "--training-kwargs",
        json.dumps(SAMPLE_TRAIN_KWARGS),
    ):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == train.INTERNAL_ERROR_EXIT_CODE
