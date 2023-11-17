"""
This sets up global test configs when pytest starts
"""


# Standard
import os
import shlex
import socket
import subprocess
import sys
import threading

# Third Party
from fixtures import Fixtures
import pytest

# Local
from caikit.core.data_model.dataobject import render_dataobject_protos
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from tests.conftest import random_test_id


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
def sample_predict_servicer(sample_inference_service) -> GlobalPredictServicer:
    servicer = GlobalPredictServicer(inference_service=sample_inference_service)
    yield servicer
    # Make sure to not leave the rpc_meter hanging
    # (It does try to clean itself up on destruction, but just to be sure)
    rpc_meter = getattr(servicer, "rpc_meter", None)
    if rpc_meter:
        rpc_meter.end_writer_thread()


@pytest.fixture(scope="session")
def sample_train_servicer(sample_train_service) -> GlobalTrainServicer:
    servicer = GlobalTrainServicer(training_service=sample_train_service)
    yield servicer


@pytest.fixture
def sample_task_unary_rpc(sample_inference_service: ServicePackage) -> TaskPredictRPC:
    return sample_inference_service.caikit_rpcs["SampleTaskPredict"]


@pytest.fixture
def sample_task_streaming_rpc(
    sample_inference_service: ServicePackage,
) -> TaskPredictRPC:
    return sample_inference_service.caikit_rpcs["ServerStreamingSampleTaskPredict"]


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
