"""
This sets up global http test configs when pytest starts
"""

# Standard
from contextlib import contextmanager
from functools import partial
from typing import Dict
import os
import tempfile
import time
import warnings

# Third Party
import pytest
import requests

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.runtime.grpc.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.http import http_server
from caikit.runtime.model_management.loaded_model import LoadedModel
from tests.conftest import temp_config

log = alog.use_channel("TEST-HTTPCONF")


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


def register_trained_model(
    servicer: GlobalPredictServicer,
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
    servicer._model_manager.loaded_models[model_id] = loaded_model


# IMPLEMENTATION DETAILS ############################################################


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
