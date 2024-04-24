"""
This sets up global test configs when pytest starts
"""

# Standard
from contextlib import closing, contextmanager
from functools import partial
from typing import Dict, List, Optional, Type, Union
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings

# Third Party
from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc
import pytest
import requests
import tls_test_tools

# First Party
import aconfig
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model.dataobject import (
    DataObjectBase,
    dataobject,
    render_dataobject_protos,
)
from caikit.runtime import http_server
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.servicers.model_runtime_servicer import ModelRuntimeServicerImpl
from caikit.runtime.work_management.abortable_context import ThreadInterrupter
from tests.conftest import random_test_id, temp_config
from tests.fixtures import Fixtures

log = alog.use_channel("TEST-CONFTEST")


def get_open_port():
    """Non-fixture function to get an open port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        return port


@pytest.fixture
def open_port():
    """Get an open port on localhost
    Returns:
        int: Available port
    """
    return get_open_port()


@pytest.fixture(scope="session")
def session_scoped_open_port():
    """Get an open port on localhost
    Returns:
        int: Available port
    """
    return get_open_port()


@pytest.fixture(scope="session")
def http_session_scoped_open_port():
    """Get an open port on localhost
    Returns:
        int: Available port
    """
    return get_open_port()


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
    interrupter = ThreadInterrupter()
    interrupter.start()
    servicer = GlobalPredictServicer(
        inference_service=sample_inference_service, interrupter=interrupter
    )
    yield servicer
    # Make sure to not leave the rpc_meter hanging
    # (It does try to clean itself up on destruction, but just to be sure)
    rpc_meter = getattr(servicer, "rpc_meter", None)
    if rpc_meter:
        rpc_meter.end_writer_thread()
    interrupter.stop()


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
def runtime_grpc_server(session_scoped_open_port) -> RuntimeGRPCServer:
    with runtime_grpc_test_server(
        session_scoped_open_port,
    ) as server:
        _check_server_readiness(server)
        yield server


@pytest.fixture(scope="session")
def model_runtime_servicer(runtime_grpc_server) -> ModelRuntimeServicerImpl:
    # Builds a new servicer, the one in the server is a bit hard to access
    return ModelRuntimeServicerImpl(interrupter=runtime_grpc_server.interrupter)


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
            # Forward the special "tls_config_override" to "tls_config_override"
            # IFF the configs contain actual TLS (indicated by the presence of
            # the special "use_in_test" element).
            config_overrides = kwargs.pop("tls_config_override", {})
            if tls_config_override := config_overrides.get("runtime", {}).get("tls"):
                kwargs["tls_config_override"] = tls_config_override
            else:
                config_overrides = {}
            check_readiness = kwargs.pop("check_readiness", True)
            with http_server.RuntimeHTTPServer(*args, **kwargs) as server:
                if check_readiness:
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


@contextmanager
def runtime_test_server(*args, protocol: str = "grpc", **kwargs):
    """Helper function to yield either server"""
    if protocol == "http":
        with runtime_http_test_server(*args, **kwargs) as server:
            yield server
    elif protocol == "grpc":
        with runtime_grpc_test_server(*args, **kwargs) as server:
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
def primitive_task_model_id(primitive_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
    model_manager = ModelManager.get_instance()
    # model load test already tests with archive - just using a model path here
    model_manager.load_model(
        model_id,
        local_model_path=primitive_model_path,
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


@pytest.fixture
def multi_task_model_id(multi_task_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
    model_manager = ModelManager.get_instance()
    # model load test already tests with archive - just using a model path here
    model_manager.load_model(
        model_id,
        local_model_path=multi_task_model_path,
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
        self.proc = subprocess.Popen(
            self._cmd, env=self._env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
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
    cert = None
    # tls
    if config_overrides:
        mode = "https"
        # mtls
        if "client_cert" and "client_key" in config_overrides["use_in_test"]:
            cert = (
                config_overrides["use_in_test"]["client_cert"],
                config_overrides["use_in_test"]["client_key"],
            )
    done = False
    while not done:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="urllib3")
                response = requests.get(
                    f"{mode}://localhost:{server.port}{http_server.HEALTH_ENDPOINT}",
                    verify=False,
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


## TLS Helpers #####################################################################


@dataobject(package="caikit_data_model.test")
class KeyPair(DataObjectBase):
    cert: str
    key: str


@dataobject(package="caikit_data_model.test")
class TLSConfig(DataObjectBase):
    server: KeyPair
    client: KeyPair


@contextmanager
def generate_tls_configs(
    port: int,
    tls: bool = False,
    mtls: bool = False,
    inline: bool = False,
    separate_client_ca: bool = False,
    server_sans: Optional[List[str]] = None,
    client_sans: Optional[List[str]] = None,
    **http_config_overrides,
) -> Dict[str, Dict]:
    """Helper to generate tls configs"""
    with tempfile.TemporaryDirectory() as workdir:
        config_overrides = {}
        client_keyfile, client_certfile = None, None
        ca_cert, server_cert, server_key = None, None, None
        use_in_test = config_overrides.setdefault("use_in_test", {})
        use_in_test["workdir"] = workdir
        if mtls or tls:
            ca_key = tls_test_tools.generate_key()[0]
            ca_cert = tls_test_tools.generate_ca_cert(ca_key)
            server_key, server_cert = tls_test_tools.generate_derived_key_cert_pair(
                ca_key=ca_key,
                san_list=server_sans,
            )
            server_certfile, server_keyfile = save_key_cert_pair(
                "server", workdir, server_key, server_cert
            )

            if inline:
                tls_config = TLSConfig(
                    server=KeyPair(cert=server_cert, key=server_key),
                    client=KeyPair(cert="", key=""),
                )
            else:
                tls_config = TLSConfig(
                    server=KeyPair(cert=server_certfile, key=server_keyfile),
                    client=KeyPair(cert="", key=""),
                )

            # need to save this ca_certfile in config_overrides so the tls
            # tests below can access it from client side
            ca_certfile, _ = save_key_cert_pair("ca", workdir, cert=ca_cert)
            use_in_test["ca_cert"] = ca_certfile
            use_in_test["server_key"] = server_keyfile
            use_in_test["server_cert"] = server_certfile

            # also saving a bad ca_certfile for a failure test case
            bad_ca_file = os.path.join(workdir, "bad_ca_cert.crt")
            with open(bad_ca_file, "w") as handle:
                bad_cert = (
                    "-----BEGIN CERTIFICATE-----\nfoobar\n-----END CERTIFICATE-----"
                )
                handle.write(bad_cert)
            use_in_test["bad_ca_cert"] = bad_ca_file

            if mtls:
                if separate_client_ca:
                    subject_kwargs = {"common_name": "my.client"}
                    client_ca_key = tls_test_tools.generate_key()[0]
                    client_ca_cert = tls_test_tools.generate_ca_cert(
                        client_ca_key, **subject_kwargs
                    )
                else:
                    subject_kwargs = {}
                    client_ca_key = ca_key
                    client_ca_cert = ca_cert

                # If inlining the client CA
                if inline:
                    tls_config.client.cert = client_ca_cert
                else:
                    client_ca_certfile, _ = save_key_cert_pair(
                        "client_ca", workdir, cert=client_ca_cert
                    )
                    tls_config.client.cert = client_ca_certfile

                # Set up the client key/cert pair derived from the client CA
                client_certfile, client_keyfile = save_key_cert_pair(
                    "client",
                    workdir,
                    *tls_test_tools.generate_derived_key_cert_pair(
                        ca_key=client_ca_key,
                        san_list=client_sans,
                        **subject_kwargs,
                    ),
                )
                # need to save the client cert and key in config_overrides so the mtls test below can access it
                use_in_test["client_cert"] = client_certfile
                use_in_test["client_key"] = client_keyfile

            config_overrides["runtime"] = {"tls": tls_config.to_dict()}
        config_overrides.setdefault("runtime", {})["http"] = {
            "server_shutdown_grace_period_seconds": 0.01,  # this is so the server is killed after 0.1 if no test is running
            "port": port,
            **http_config_overrides,
        }

        with temp_config(config_overrides, "merge"):
            yield aconfig.Config(config_overrides)


def save_key_cert_pair(prefix, workdir, key=None, cert=None):
    crtfile, keyfile = None, None
    if key is not None:
        keyfile = os.path.join(workdir, f"{prefix}.key")
        with open(keyfile, "w") as handle:
            handle.write(key)
    if cert is not None:
        crtfile = os.path.join(workdir, f"{prefix}.crt")
        with open(crtfile, "w") as handle:
            handle.write(cert)
    return crtfile, keyfile


@pytest.fixture
def deploy_good_model_files():
    model_files = {}
    model_path = Fixtures.get_good_model_path()
    for fname in os.listdir(model_path):
        with open(os.path.join(model_path, fname), "rb") as handle:
            model_files[fname] = handle.read()
    yield model_files
