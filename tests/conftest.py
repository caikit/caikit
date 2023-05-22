"""
This sets up global test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from typing import Type
from unittest.mock import patch
import copy
import json
import os
import sys
import tempfile
import threading
import time
import uuid

# Third Party
from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc
import pytest

# First Party
import alog

# Local
from caikit import get_config
from caikit.core.data_model.dataobject import render_dataobject_protos
from caikit.core.toolkit import logging
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from tests.fixtures import Fixtures
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
    yield
    # No cleanup required...?


@pytest.fixture(scope="session")
def sample_inference_service(render_protos) -> ServicePackage:
    """Service package pointing to `sample_lib` for testing"""
    inference_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
        ServicePackageFactory.ServiceSource.GENERATED,
    )
    if render_protos:
        render_dataobject_protos("tests/protos")
    return inference_service


@pytest.fixture(scope="session")
def sample_predict_servicer(sample_inference_service) -> GlobalPredictServicer:
    servicer = GlobalPredictServicer(inference_service=sample_inference_service)
    yield servicer
    # Make sure to not leave the rpc_meter hanging
    # (It does try to clean itself up on destruction, but just to be sure)
    servicer.rpc_meter.end_writer_thread()


@pytest.fixture(scope="session")
def sample_train_service(render_protos) -> ServicePackage:
    """Service package pointing to `sample_lib` for testing"""
    training_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
        ServicePackageFactory.ServiceSource.GENERATED,
    )
    if render_protos:
        render_dataobject_protos("tests/protos")
    return training_service


@pytest.fixture(scope="session")
def sample_train_servicer(sample_train_service) -> GlobalTrainServicer:
    servicer = GlobalTrainServicer(training_service=sample_train_service)
    yield servicer


@pytest.fixture(scope="session")
def runtime_grpc_server(
    sample_inference_service, sample_train_service
) -> RuntimeGRPCServer:
    server = RuntimeGRPCServer(
        inference_service=sample_inference_service,
        training_service=sample_train_service,
    )

    grpc_thread = threading.Thread(
        target=server.start,
    )
    grpc_thread.setDaemon(False)
    grpc_thread.start()
    _check_server_readiness(server)
    yield server

    # teardown
    server.stop(0)
    grpc_thread.join()


@pytest.fixture(scope="session")
def inference_stub(sample_inference_service, runtime_grpc_server) -> Type:
    inference_stub = sample_inference_service.stub_class(
        runtime_grpc_server.make_local_channel()
    )
    return inference_stub


@pytest.fixture(scope="session")
def train_stub(sample_train_service, runtime_grpc_server) -> Type:
    train_stub = sample_train_service.stub_class(
        runtime_grpc_server.make_local_channel()
    )
    return train_stub


@pytest.fixture
def good_model_path() -> str:
    return Fixtures.get_good_model_path()


@pytest.fixture
def other_good_model_path() -> str:
    return Fixtures.get_other_good_model_path()


@pytest.fixture
def loaded_model_id(good_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = _random_id()
    model_manager = ModelManager.get_instance()
    # model load test already tests with archive - just using a model path here
    model_manager.load_model(
        model_id,
        local_model_path=good_model_path,
        model_type=Fixtures.get_good_model_type(),  # eventually we'd like to be determining the type from the model itself...
    )
    yield model_id

    # teardown
    model_manager.unload_model(model_id)


@pytest.fixture
def other_loaded_model_id(other_good_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = _random_id()
    model_manager = ModelManager.get_instance()
    # model load test already tests with archive - just using a model path here
    model_manager.load_model(
        model_id,
        local_model_path=other_good_model_path,
        model_type=Fixtures.get_good_model_type(),  # eventually we'd like to be determining the type from the model itself...
    )
    yield model_id

    # teardown
    model_manager.unload_model(model_id)


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
    parser.addoption(
        "--render-protos",
        action="store_true",
        default=False,
        help="Render test protos for debug?",
    )


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
    yield os.path.join(os.path.dirname(os.path.realpath(__file__)), "fixtures")


@pytest.fixture
def modules_fixtures_dir(fixtures_dir):
    yield os.path.join(fixtures_dir, "modules")


@pytest.fixture
def toolkit_fixtures_dir(fixtures_dir):
    yield os.path.join(fixtures_dir, "toolkit")


# IMPLEMENTATION DETAILS ############################################################
def _random_id():
    return str(uuid.uuid4())


def _check_server_readiness(server):
    """Check server readiness"""

    channel = grpc.insecure_channel(f"localhost:{server.port}")

    done = False
    while not done:
        try:
            stub = health_pb2_grpc.HealthStub(channel)
            health_check_request = health_pb2.HealthCheckRequest()
            stub.Check(health_check_request)
            done = True
        except grpc.RpcError:
            log.debug("[RpcError]; will try to reconnect to test server in 0.1 second.")
            time.sleep(0.1)
