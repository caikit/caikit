"""
This sets up global test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from typing import Callable, List, Union
from unittest.mock import patch
import copy
import importlib
import os
import platform
import sys
import tempfile
import uuid

# Third Party
import pytest
import semver

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import MODEL_MANAGER
from caikit.core.registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
    module_registry,
)
from caikit.core.toolkit import logging
import caikit

log = alog.use_channel("TEST-CONFTEST")

FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
)

# Some tests need to be skipped if using protobuf 3.X and arm
PROTOBUF_VERSION = semver.parse(importlib.metadata.version("protobuf"))["major"]
ARM_ARCH = "arm" in platform.machine()

# Make sample_lib available for import
sys.path.append(FIXTURES_DIR)
# Local
import sample_lib

# Configure logging from the environment
logging.configure()


def random_test_id():
    return "test-any-model-" + _random_id()


@pytest.fixture(autouse=True, scope="session")
def test_environment():
    """The most important fixture: This runs caikit configuration with the base test config overrides"""
    test_config_path = os.path.join(FIXTURES_DIR, "config", "config.yml")
    caikit.configure(test_config_path)
    with tempfile.TemporaryDirectory() as workdir:
        # 🌶️ All tests need to clean up after themselves! The metering and
        # training output directories are the one thing easily creates leftover
        # files, so we explicitly update config here to point them at a temp dir
        with temp_config(
            {
                "runtime": {
                    "metering": {
                        "log_dir": os.path.join(workdir, "metering_logs"),
                    },
                    "training": {
                        "output_dir": os.path.join(workdir, "training_output"),
                    },
                }
            },
            "merge",
        ):
            # import the mock backend that is specified in the config
            # This is required to run any test that loads a model
            # Local
            from tests.core.helpers import MockBackend

            yield
    # No cleanup required...?


@pytest.fixture
def good_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "models", "foo")


@pytest.fixture
def box_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "models", "box")


@pytest.fixture
def primitive_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "models", "primitive")


@pytest.fixture
def streaming_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "dummy_streaming_module")


@pytest.fixture
def other_good_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "models", "bar")


@pytest.fixture
def multi_task_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "models", "multi")


# Sample data files for testing ###########################
@pytest.fixture
def data_stream_inputs() -> str:
    return os.path.join(FIXTURES_DIR, "data_stream_inputs")


@pytest.fixture
def sample_json_file(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample.json")


@pytest.fixture
def sample_json_collection(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample_json_collection")


@pytest.fixture
def sample_csv_file(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample_w_header.csv")


@pytest.fixture
def sample_csv_file_no_headers(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample.csv")


@pytest.fixture
def sample_csv_collection(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample_csv_collection")


@pytest.fixture
def sample_multipart_json(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "multi_part_json.txt")


@pytest.fixture
def sample_multipart_json_with_content_header(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "multi_part_json_with_content_header.txt")


@pytest.fixture
def sample_multipart_csv(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "multi_part_csv.txt")


@pytest.fixture
def sample_jsonl_file(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample.jsonl")


@pytest.fixture
def sample_jsonl_collection(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample_jsonl_collection")


@pytest.fixture
def jsonl_with_control_chars(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "control_chars.jsonl")


@pytest.fixture
def sample_text_file(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample.txt")


@pytest.fixture
def sample_text_collection(data_stream_inputs) -> str:
    return os.path.join(data_stream_inputs, "sample_txt_collection")


@contextmanager
def temp_config(config_overrides: dict, merge_strategy="override"):
    """Temporarily edit the caikit config in a mock context"""
    existing_config = copy.deepcopy(getattr(caikit.config.config, "_CONFIG"))
    # Patch out the internal config, starting with a fresh copy of the current config
    with patch.object(caikit.config.config, "_CONFIG", existing_config):
        # Patch the immutable view of the config as well
        # This is required otherwise the updated immutable view will persist after the test
        with patch.object(caikit.config.config, "_IMMUTABLE_CONFIG", None):
            # Run our config overrides inside the patch
            if config_overrides:
                config_overrides["merge_strategy"] = merge_strategy
                caikit.configure(config_dict=config_overrides)
            else:
                # or just slap some random uuids in there. Barf, but we need to call `.configure()`
                caikit.configure(config_dict={str(uuid.uuid4()): str(uuid.uuid4())})
            # Yield to the test with the new overridden config
            yield get_config()


@contextmanager
def set_use_subprocess(use_subprocess: bool):
    with temp_config(
        {
            "model_management": {
                "trainers": {
                    "default": {
                        "config": {
                            "use_subprocess": use_subprocess,
                        }
                    }
                }
            }
        },
        "merge",
    ):
        yield


# fixtures to optionally generate the protos for easier debugging
def pytest_addoption(parser):
    try:
        parser.addoption(
            "--render-protos",
            action="store_true",
            default=False,
            help="Render test protos for debug?",
        )
    except ValueError:
        pass


@pytest.fixture(scope="session")
def render_protos(request):
    return request.config.getoption("--render-protos")


@pytest.fixture
def sample_int_file() -> str:
    content = "[1,2,3,4,5,6,7,8,9,10]"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
        handle.write(content)
        handle.flush()
        yield handle.name


@pytest.fixture
def fixtures_dir():
    yield FIXTURES_DIR


@pytest.fixture
def modules_fixtures_dir(fixtures_dir):
    yield os.path.join(fixtures_dir, "modules")


@pytest.fixture
def toolkit_fixtures_dir(fixtures_dir):
    yield os.path.join(fixtures_dir, "toolkit")


@contextmanager
def backend_priority(backend_cfg: Union[List[dict], dict]):
    if isinstance(backend_cfg, dict):
        backend_cfg = [backend_cfg]
    with temp_config(
        {
            "model_management": {
                "finders": {"default": {"type": "LOCAL"}},
                "initializers": {
                    "default": {
                        "type": "LOCAL",
                        "config": {"backend_priority": backend_cfg},
                    }
                },
            }
        }
    ):
        yield


class TempFailWrapper:
    """Helper that can wrap a callable with a sequence of failures"""

    def __init__(
        self,
        func: Callable,
        num_failures: int = 1,
        exc: Exception = RuntimeError("Yikes"),
    ):
        self.func = func
        self.num_failures = num_failures
        self.exc = exc

    def __call__(self, *args, **kwargs):
        if self.num_failures:
            self.num_failures -= 1
            raise self.exc
        return self.func(*args, **kwargs)


# IMPLEMENTATION DETAILS ############################################################
def _random_id():
    return str(uuid.uuid4())


@pytest.fixture
def reset_backend_types():
    """Fixture that will reset the backend types if a test modifies them"""
    base_backend_types = {key: val for key, val in module_backend_types().items()}
    base_backend_classes = {key: val for key, val in module_backend_classes().items()}
    yield
    module_backend_types().clear()
    module_backend_types().update(base_backend_types)
    module_backend_classes().clear()
    module_backend_classes().update(base_backend_classes)


@pytest.fixture
def reset_module_backend_registry():
    """Fixture that will reset the module distribution registry if a test modifies them"""
    # NB: Deepcopy here because the values are nested dicts that will be persisted by reference
    orig_module_backend_registry = copy.deepcopy(module_backend_registry())
    yield
    module_backend_registry().clear()
    module_backend_registry().update(orig_module_backend_registry)


@pytest.fixture
def reset_module_registry():
    """Fixture that will reset caikit.core module registry if a test modifies it"""
    orig_module_registry = {key: val for key, val in module_registry().items()}
    yield
    module_registry().clear()
    module_registry().update(orig_module_registry)


@pytest.fixture
def reset_model_manager():
    prev_finders = MODEL_MANAGER._finders
    prev_initializers = MODEL_MANAGER._initializers
    prev_trainers = MODEL_MANAGER._trainers
    MODEL_MANAGER._finders = {}
    MODEL_MANAGER._initializers = {}
    MODEL_MANAGER._trainers = {}
    yield
    MODEL_MANAGER._finders = prev_finders
    MODEL_MANAGER._initializers = prev_initializers
    MODEL_MANAGER._trainers = prev_trainers


@pytest.fixture
def reset_globals(
    reset_backend_types,
    reset_model_manager,
    reset_module_backend_registry,
    reset_module_registry,
):
    """Fixture that will reset the backend types and module registries if a test modifies them"""
