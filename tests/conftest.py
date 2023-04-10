"""
This sets up global test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from typing import Type
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
import yaml

# First Party
import alog

# Local
from caikit.core.data_model.dataobject import render_dataobject_protos
from caikit.core.toolkit import logging
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.model_loader import ModelLoader
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.utils.config_parser import ConfigParser
from tests.fixtures import Fixtures

log = alog.use_channel("TEST-CONFTEST")

# Make sample_lib available for import
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "fixtures",
    ),
)

# Configure logging from the environment
logging.configure(
    default_level=os.environ.get("LOG_LEVEL", "off"),
    filters=os.environ.get("LOG_FILTERS", "urllib3:off"),
    thread_id=os.environ.get("LOG_THREAD_ID", "") == "true",
)


@pytest.fixture(autouse=True, scope="session")
def test_environment():
    """The most important fixture: This sets `ENVIRONMENT=test` for all tests.
    This is required to pick up the `test` section of config so that our unit
    tests pick up the correct settings.
    """
    old_env = os.environ.get("ENVIRONMENT")
    os.environ["ENVIRONMENT"] = "test"
    # hack: delete any config that exists
    ConfigParser._ConfigParser__instance = None
    cfg = ConfigParser.get_instance()
    # Make sure we picked up teh test configs
    assert cfg.environment == "test"
    # Test away!
    yield
    # Reset environment back to previous state
    if old_env is None:
        os.unsetenv("ENVIRONMENT")
    else:
        os.environ["ENVIRONMENT"] = old_env


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
        infer_srv=sample_inference_service,
        train_srv=sample_train_service,
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
def temp_config_parser(config_overrides):
    """Temporarily overwrite the ConfigParser singleton"""
    real_singleton = ConfigParser.get_instance()
    prev_config_path = os.environ.get("CONFIG_FILES")
    ConfigParser._ConfigParser__instance = None
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as temp_cfg:
        yaml.safe_dump(config_overrides, temp_cfg)
        temp_cfg.flush()
        os.environ["CONFIG_FILES"] = temp_cfg.name
        temp_inst = ConfigParser.get_instance()
        ModelLoader.get_instance().config_parser = temp_inst
        yield temp_inst
    ConfigParser._ConfigParser__instance = real_singleton
    ModelLoader.get_instance().config_parser = ConfigParser.get_instance()
    if prev_config_path is None:
        del os.environ["CONFIG_FILES"]
    else:
        os.environ["CONFIG_FILES"] = prev_config_path


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
