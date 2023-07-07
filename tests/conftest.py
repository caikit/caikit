"""
This sets up global test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from typing import List, Union
from unittest.mock import patch
import copy
import json
import os
import sys
import tempfile
import uuid

# Third Party
import pytest

# First Party
import alog

# Local
from caikit import get_config
from caikit.core.toolkit import logging
import caikit

log = alog.use_channel("TEST-CONFTEST")

FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
)

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
def streaming_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "dummy_streaming_module")


@pytest.fixture
def other_good_model_path() -> str:
    return os.path.join(FIXTURES_DIR, "models", "bar")


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
def sample_json_file() -> str:
    json_content = json.dumps(
        [
            {"number": 1},
            {"number": 2},
        ]
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
        handle.write(json_content)
        handle.flush()
        yield handle.name


@pytest.fixture
def sample_csv_file() -> str:
    csv_header = "number"
    csv_content = []
    csv_content.append("1")
    csv_content.append("2")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as handle:
        handle.write(csv_header)
        handle.write("\n")
        for row in csv_content:
            handle.write(row)
            handle.write("\n")
        handle.flush()
        yield handle.name


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


# IMPLEMENTATION DETAILS ############################################################
def _random_id():
    return str(uuid.uuid4())
