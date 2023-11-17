"""
This sets up global grpc test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from functools import partial
from typing import Type, Union
import os
import tempfile
import time

# Third Party
from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc
import pytest

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model.dataobject import render_dataobject_protos
from caikit.runtime.grpc_server.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from tests.conftest import temp_config

log = alog.use_channel("TEST-GRPCCONF")


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
