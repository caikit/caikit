"""
This sets up global test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from functools import partial
from typing import Dict, Type, Union
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time

# Third Party
from fastapi.testclient import TestClient
from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc
import pytest
import requests

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model.dataobject import render_dataobject_protos
from caikit.runtime import http_server
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from tests.conftest import random_test_id, temp_config
from tests.fixtures import Fixtures

log = alog.use_channel("TEST-CONFTEST")


@pytest.fixture
def open_port():
    """Get an open port on localhost
    Returns:
        int: Available port
    """
    return _open_port()


@pytest.fixture(scope="session")
def session_scoped_open_port():
    """Get an open port on localhost
    Returns:
        int: Available port
    """
    return _open_port()


@pytest.fixture(scope="session")
def http_session_scoped_open_port():
    """Get an open port on localhost
    Returns:
        int: Available port
    """
    return _open_port()


def _open_port():
    # TODO: This has obvious problems where the port returned for use by a test is not immediately
    # put into use, so parallel tests could attempt to use the same port.
    start = 8888
    end = start + 1000
    host = "localhost"
    for port in range(start, end):
        with socket.socket() as soc:
            # soc.connect_ex returns 0 if connection is successful,
            # indicating the port is in use
            if soc.connect_ex((host, port)) != 0:
                # So a non-zero code should mean the port is not currently in use
                return port


@pytest.fixture(scope="session")
def sample_inference_service(render_protos) -> ServicePackage:
    """Service package pointing to `sample_lib` for testing"""
    inference_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
    )
    if render_protos:
        output_dir = os.path.join("tests", "protos")
        render_dataobject_protos(output_dir)
        inference_service.service.write_proto_file(output_dir)
    return inference_service


@pytest.fixture(scope="session")
def sample_predict_servicer(sample_inference_service) -> GlobalPredictServicer:
    servicer = GlobalPredictServicer(inference_service=sample_inference_service)
    yield servicer
    # Make sure to not leave the rpc_meter hanging
    # (It does try to clean itself up on destruction, but just to be sure)
    rpc_meter = getattr(servicer, "rpc_meter", None)
    if rpc_meter:
        rpc_meter.end_writer_thread()


@pytest.fixture(scope="session")
def sample_train_service(render_protos) -> ServicePackage:
    """Service package pointing to `sample_lib` for testing"""
    training_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
    )
    if render_protos:
        output_dir = os.path.join("tests", "protos")
        render_dataobject_protos(output_dir)
        training_service.service.write_proto_file(output_dir)
    return training_service


@pytest.fixture(scope="session")
def sample_train_servicer(sample_train_service) -> GlobalTrainServicer:
    servicer = GlobalTrainServicer(training_service=sample_train_service)
    yield servicer


@contextmanager
def runtime_grpc_test_server(open_port, *args, **kwargs):
    """Helper to wrap creation of RuntimeGRPCServer in temporary configurations"""
    with tempfile.TemporaryDirectory() as workdir:
        temp_log_dir = os.path.join(workdir, "metering_logs")
        temp_save_dir = os.path.join(workdir, "training_output")
        os.makedirs(temp_log_dir)
        os.makedirs(temp_save_dir)
        with temp_config(
            {
                "runtime": {
                    "metering": {"log_dir": temp_log_dir},
                    "training": {"output_dir": temp_save_dir},
                    "grpc": {"port": open_port},
                }
            },
            "merge",
        ):
            with RuntimeGRPCServer(*args, **kwargs) as server:
                # Give tests access to the workdir
                server.workdir = workdir
                yield server


@pytest.fixture(scope="session")
def runtime_grpc_server(
    session_scoped_open_port,
) -> RuntimeGRPCServer:
    with runtime_grpc_test_server(
        session_scoped_open_port,
    ) as server:
        _check_server_readiness(server)
        yield server


@contextmanager
def runtime_http_test_server(open_port, *args, **kwargs):
    """Helper to wrap creation of RuntimeHTTPServer in temporary configurations"""
    with tempfile.TemporaryDirectory() as workdir:
        temp_log_dir = os.path.join(workdir, "metering_logs")
        temp_save_dir = os.path.join(workdir, "training_output")
        os.makedirs(temp_log_dir)
        os.makedirs(temp_save_dir)
        with temp_config(
            {
                "runtime": {
                    "metering": {"log_dir": temp_log_dir},
                    "training": {"output_dir": temp_save_dir},
                    "http": {"port": open_port},
                }
            },
            "merge",
        ):
            config_overrides = {}
            if "tls_config_override" in kwargs:
                config_overrides = kwargs["tls_config_override"]
                kwargs["tls_config_override"] = config_overrides["runtime"]["tls"]
            with http_server.RuntimeHTTPServer(*args, **kwargs) as server:
                _check_http_server_readiness(server, config_overrides)
                # Give tests access to the workdir
                server.workdir = workdir
                yield server


# For tests that need a working http server
# For other http tests, we can use FastAPI TestClient
@pytest.fixture(scope="session")
def runtime_http_server(
    http_session_scoped_open_port,
) -> http_server.RuntimeHTTPServer:
    """yields an actual running http server"""
    with runtime_http_test_server(
        http_session_scoped_open_port,
    ) as server:
        yield server


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


@pytest.fixture(scope="session")
def training_management_stub(runtime_grpc_server) -> Type:
    training_management_service: ServicePackage = (
        ServicePackageFactory().get_service_package(
            ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
        )
    )

    training_management_stub = training_management_service.stub_class(
        runtime_grpc_server.make_local_channel()
    )
    return training_management_stub


@pytest.fixture
def sample_task_model_id(good_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
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
def file_task_model_id(box_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
    model_manager = ModelManager.get_instance()
    # model load test already tests with archive - just using a model path here
    model_manager.load_model(
        model_id,
        local_model_path=box_model_path,
        model_type=Fixtures.get_good_model_type(),  # eventually we'd like to be determining the type from the model itself...
    )
    yield model_id

    # teardown
    model_manager.unload_model(model_id)


@pytest.fixture
def sample_task_unary_rpc(sample_inference_service: ServicePackage) -> TaskPredictRPC:
    return sample_inference_service.caikit_rpcs["SampleTaskPredict"]


@pytest.fixture
def sample_task_streaming_rpc(
    sample_inference_service: ServicePackage,
) -> TaskPredictRPC:
    return sample_inference_service.caikit_rpcs["ServerStreamingSampleTaskPredict"]


@pytest.fixture
def streaming_task_model_id(streaming_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
    model_manager = ModelManager.get_instance()
    model_manager.load_model(
        model_id,
        local_model_path=streaming_model_path,
        model_type=Fixtures.get_good_model_type(),
    )
    yield model_id

    # teardown
    model_manager.unload_model(model_id)


@pytest.fixture
def other_task_model_id(other_good_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
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


def register_trained_model(
    servicer: Union[RuntimeGRPCServer, GlobalPredictServicer],
    model_id: str,
    training_id: str,
):
    """Helper to auto-load a model that has completed training. This replaces
    the old auto-load feature which was only needed for unit tests
    """
    model_future_factory = partial(MODEL_MANAGER.get_model_future, training_id)
    loaded_model = (
        LoadedModel.Builder()
        .id(model_id)
        .type("trained")
        .path("")
        .model_future_factory(model_future_factory)
        .build()
    )
    if isinstance(servicer, RuntimeGRPCServer):
        servicer = servicer._global_predict_servicer
    servicer._model_manager.loaded_models[model_id] = loaded_model


class ModuleSubproc:
    def __init__(
        self, module_to_run: str, *args, kill_timeout: float = 10.0, **env_vars
    ):
        """Run the given python as a subprocess and kill it after the given timeout"""
        # Set up the command
        cmd = f"{sys.executable} -m {module_to_run}"
        for arg in args:
            cmd += f" {arg}"
        self._cmd = shlex.split(cmd)

        # Set up the environment
        self._env = {**os.environ, **env_vars}
        self._env["PYTHONPATH"] = ":".join(sys.path)

        # Start the process
        self.proc = None
        self._killed = False

        # Start the timer that will kill the process
        self._kill_timer = threading.Timer(kill_timeout, self._kill_proc)

    @property
    def killed(self):
        return self._killed

    def _kill_proc(self):
        self._killed = True
        if self.proc is not None:
            self.proc.kill()

    def __enter__(self):
        self.proc = subprocess.Popen(self._cmd, env=self._env)
        self._kill_timer.start()
        return self.proc

    def __exit__(self, *_):
        self.proc.wait()
        self._kill_timer.cancel()
        self._kill_timer.join()


# IMPLEMENTATION DETAILS ############################################################


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
            log.debug(
                "[RpcError]; will try to reconnect to test server in 0.01 second."
            )
            time.sleep(0.01)


def _check_http_server_readiness(server, config_overrides: Dict[str, Dict]):
    mode = "http"
    verify = None
    cert = None
    # tls
    if config_overrides:
        mode = "https"
        verify = config_overrides["use_in_test"]["ca_cert"]
        # mtls
        if "client_cert" and "client_key" in config_overrides["use_in_test"]:
            cert = (
                config_overrides["use_in_test"]["client_cert"],
                config_overrides["use_in_test"]["client_key"],
            )
    done = False
    while not done:
        try:
            response = requests.get(
                f"{mode}://localhost:{server.port}{http_server.HEALTH_ENDPOINT}",
                verify=verify,
                cert=cert,
            )
            assert response.status_code == 200
            assert response.text == "OK"
            done = True
        except AssertionError:
            log.debug(
                "[HTTP server not ready]; will try to reconnect to test server in 0.01 second."
            )
            time.sleep(0.001)
