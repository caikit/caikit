# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the caikit HTTP server
"""
# Standard
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterable
from unittest import mock
import base64
import json
import os
import signal
import tempfile
import zipfile

# Third Party
from fastapi.testclient import TestClient
import pytest
import requests

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model import TrainingStatus
from caikit.core.model_management.multi_model_finder import MultiModelFinder
from caikit.runtime import http_server
from caikit.runtime.http_server.http_server import StreamEventTypes
from caikit.runtime.server_base import ServerThreadPool
from tests.conftest import temp_config
from tests.fixtures import Fixtures
from tests.runtime.conftest import (
    ModuleSubproc,
    deploy_good_model_files,
    generate_tls_configs,
    register_trained_model,
    runtime_http_test_server,
)
from tests.runtime.model_management.test_model_manager import (
    non_singleton_model_managers,
)

################################################################################
# NOTE for test authors:
#
# This test module is quite large. Please write tests under the appropriate
# header section so that tests can be more easily discovered and managed as the
# test suite grows.
################################################################################

## Fixtures ####################################################################


@contextmanager
def client_context(server) -> Iterable[TestClient]:
    with TestClient(server.app) as client:
        yield client


@pytest.fixture
def client(runtime_http_server) -> Iterable[TestClient]:
    with client_context(runtime_http_server) as client:
        yield client


## Insecure and TLS Tests ######################################################


def test_insecure_server(runtime_http_server, open_port):
    with generate_tls_configs(open_port):
        # start a non-blocking http server
        resp = requests.get(f"http://localhost:{runtime_http_server.port}/docs")
        resp.raise_for_status()


def test_basic_tls_server(open_port):
    with generate_tls_configs(open_port, tls=True, mtls=False) as config_overrides:
        with runtime_http_test_server(
            open_port,
            tls_config_override=config_overrides,
        ) as http_server_with_tls:
            # start a non-blocking http server with basic tls
            resp = requests.get(
                f"https://localhost:{http_server_with_tls.port}/docs",
                verify=config_overrides["use_in_test"]["ca_cert"],
            )
            resp.raise_for_status()


def test_basic_tls_server_with_wrong_cert(open_port):
    with generate_tls_configs(open_port, tls=True, mtls=False) as config_overrides:
        with runtime_http_test_server(
            open_port,
            tls_config_override=config_overrides,
        ) as http_server_with_tls:
            # start a non-blocking http server with basic tls
            with pytest.raises(requests.exceptions.SSLError):
                requests.get(
                    f"https://localhost:{http_server_with_tls.port}/docs",
                    verify=config_overrides["use_in_test"]["bad_ca_cert"],
                )


def test_mutual_tls_server(open_port):
    with generate_tls_configs(open_port, tls=True, mtls=True) as config_overrides:
        with runtime_http_test_server(
            open_port,
            tls_config_override=config_overrides,
        ) as http_server_with_mtls:
            # start a non-blocking http server with mutual tls
            resp = requests.get(
                f"https://localhost:{http_server_with_mtls.port}/docs",
                verify=config_overrides["use_in_test"]["ca_cert"],
                cert=(
                    config_overrides["use_in_test"]["client_cert"],
                    config_overrides["use_in_test"]["client_key"],
                ),
            )
            resp.raise_for_status()


def test_mutual_tls_server_different_client_ca(open_port):
    with generate_tls_configs(
        open_port,
        tls=True,
        mtls=True,
        separate_client_ca=True,
    ) as config_overrides:
        # start a non-blocking http server with mutual tls
        with runtime_http_test_server(
            open_port,
            tls_config_override=config_overrides,
        ) as http_server_with_mtls:
            # Make a request with the client's key/cert pair
            resp = requests.get(
                f"https://localhost:{http_server_with_mtls.port}/docs",
                verify=config_overrides["use_in_test"]["ca_cert"],
                cert=(
                    config_overrides["use_in_test"]["client_cert"],
                    config_overrides["use_in_test"]["client_key"],
                ),
            )
            resp.raise_for_status()


def test_mutual_tls_server_inline(open_port):
    """Test that mutual TLS works when the TLS content is passed by value rather
    than with files
    """
    with generate_tls_configs(
        open_port, tls=True, mtls=True, inline=True
    ) as config_overrides:
        with runtime_http_test_server(
            open_port,
            tls_config_override=config_overrides,
        ) as http_server_with_mtls:
            # start a non-blocking http server with mutual tls
            resp = requests.get(
                f"https://localhost:{http_server_with_mtls.port}/docs",
                verify=config_overrides["use_in_test"]["ca_cert"],
                cert=(
                    config_overrides["use_in_test"]["client_cert"],
                    config_overrides["use_in_test"]["client_key"],
                ),
            )
            resp.raise_for_status()


def test_mutual_tls_server_with_wrong_cert(open_port):
    with generate_tls_configs(open_port, tls=True, mtls=True) as config_overrides:
        with runtime_http_test_server(
            open_port,
            tls_config_override=config_overrides,
        ) as http_server_with_mtls:
            # start a non-blocking http server with mutual tls
            with pytest.raises(requests.exceptions.SSLError):
                requests.get(
                    f"https://localhost:{http_server_with_mtls.port}/docs",
                    verify=config_overrides["use_in_test"]["ca_cert"],
                    cert=(
                        config_overrides["use_in_test"]["client_key"],
                        config_overrides["use_in_test"]["client_cert"],
                    ),  # flip the order of key and cert, this will result in SSLError
                )


@pytest.mark.parametrize(
    ["enable_inference", "enable_training"],
    [(True, False), (False, True), (False, False)],
)
def test_services_disabled(open_port, enable_inference, enable_training):
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "enable_inference": enable_inference,
                    "enable_training": enable_training,
                }
            },
        },
        "merge",
    ):
        with runtime_http_test_server(open_port) as server:
            # start a non-blocking http server with basic tls
            resp = requests.get(
                f"http://localhost:{open_port}{http_server.HEALTH_ENDPOINT}",
            )
            resp.raise_for_status()
            assert server.enable_inference == enable_inference
            assert (
                server.global_predict_servicer
                and server.model_management_servicer
                and enable_inference
            ) or (
                server.global_predict_servicer is None
                and server.model_management_servicer is None
                and not enable_inference
            )
            assert server.enable_training == enable_training
            assert (
                server.global_train_servicer
                and server.training_management_servicer
                and enable_training
            ) or (
                server.global_train_servicer is None
                and server.training_management_servicer is None
                and not enable_training
            )


@pytest.mark.parametrize(
    ["config_overrides", "expected"],
    [
        ({}, None),
        ({"runtime": {"http": {"server_config": {"limit_concurrency": 0}}}}, None),
        ({"runtime": {"http": {"server_config": {"limit_concurrency": 123}}}}, 123),
        (
            {
                "runtime": {
                    "server_thread_pool_size": 4,
                    "http": {"server_config": {"limit_concurrency": -1}},
                }
            },
            8,
        ),
    ],
)
def test_http_server_concurrency_limiting(config_overrides, expected):
    """Make sure that when the config for limiting concurrency is set, it is
    correctly parsed when initializing the server
    """
    with temp_config(config_overrides, merge_strategy="merge"):
        with mock.patch.object(
            ServerThreadPool, "pool", ServerThreadPool._build_pool()
        ):
            svr = http_server.RuntimeHTTPServer()
            assert svr.server.config.limit_concurrency == expected


## Lifecycle Tests #############################################################


def test_http_server_shutdown_with_model_poll(open_port):
    """Test that a SIGINT successfully shuts down the running server"""
    with tempfile.TemporaryDirectory() as workdir:
        server_proc = ModuleSubproc(
            "caikit.runtime.http_server",
            RUNTIME_HTTP_PORT=str(open_port),
            RUNTIME_LOCAL_MODELS_DIR=workdir,
            RUNTIME_LAZY_LOAD_LOCAL_MODELS="true",
            RUNTIME_LAZY_LOAD_POLL_PERIOD_SECONDS="0.1",
        )
        with server_proc as proc:
            # Wait for the server to be up
            while True:
                try:
                    resp = requests.get(
                        f"http://localhost:{open_port}{http_server.HEALTH_ENDPOINT}",
                        timeout=0.1,
                    )
                    resp.raise_for_status()
                    break
                except (
                    requests.HTTPError,
                    requests.ConnectionError,
                    requests.ConnectTimeout,
                ):
                    pass

            # Signal the server to shut down
            proc.send_signal(signal.SIGINT)

        # Make sure the process was not killed
        assert not server_proc.killed


def test_http_and_grpc_server_share_threadpool(
    runtime_http_server, runtime_grpc_server
):
    """Test that the grpc server and http server share a common thread pool"""
    assert runtime_grpc_server.thread_pool is runtime_http_server.thread_pool


def test_docs(client):
    """Simple check that pinging /docs returns 200"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_docs_using_running_http_server(runtime_http_server):
    """Simple check that pinging /docs returns 200
    but pints the actual running server"""
    response = requests.get(f"http://localhost:{runtime_http_server.port}/docs")
    assert response.status_code == 200


def test_docs_with_models(
    runtime_http_server, sample_task_model_id, primitive_task_model_id
):
    """Simple check that pinging /docs still returns 200 when models have been
    loaded"""
    response = requests.get(f"http://localhost:{runtime_http_server.port}/docs")
    assert response.status_code == 200


def test_uvicorn_server_config_valid():
    """Make sure that arbitrary uvicorn configs can be passed through from
    runtime.http.server_config
    """
    timeout_keep_alive = 10
    with temp_config(
        {
            "runtime": {
                "http": {"server_config": {"timeout_keep_alive": timeout_keep_alive}}
            }
        },
        "merge",
    ):
        server = http_server.RuntimeHTTPServer()
        assert server.server.config.timeout_keep_alive == timeout_keep_alive


def test_uvicorn_server_config_invalid_tls_overlap():
    """Make sure uvicorn TLS arguments cannot be set if TLS is enabled in caikit
    config
    """
    with temp_config(
        {
            "runtime": {
                "http": {
                    "server_config": {
                        "ssl_keyfile": "/some/file.pem",
                    }
                }
            }
        },
        "merge",
    ):
        with generate_tls_configs(port=1234, tls=True, mtls=True):
            with pytest.raises(ValueError):
                http_server.RuntimeHTTPServer()


def test_uvicorn_server_config_invalid_kwarg_overlap():
    """Make sure uvicorn config can't be set for configs that caikit manages"""
    with temp_config(
        {
            "runtime": {
                "http": {
                    "server_config": {
                        "log_level": "debug",
                    }
                }
            }
        },
        "merge",
    ):
        with pytest.raises(ValueError):
            http_server.RuntimeHTTPServer()


## Inference Tests #############################################################


def test_inference_sample_task(sample_task_model_id, client):
    """Simple check that we can ping a model"""
    json_input = {"inputs": {"name": "world"}, "model_id": sample_task_model_id}
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["greeting"] == "Hello world"


def test_inference_primitive_task(primitive_task_model_id, client):
    """Simple check that we can ping a model"""
    json_input = {
        "inputs": {"name": "hello"},
        "parameters": {
            "bool_type": True,
            "int_type": 1,
            "float_type": 1.0,
            "str_type": "astring",
            "bytes_type": "cmF3Ynl0ZXMK",
            "list_type": ["list", "of", "strings"],
        },
        "model_id": primitive_task_model_id,
    }
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert "hello: primitives!" in json_response["greeting"]


def test_inference_sample_task_optional_field(sample_task_model_id, client):
    """Simple check for optional fields"""
    json_input = {
        "model_id": sample_task_model_id,
        "inputs": {"name": "world"},
        "parameters": {"throw": True},
    }
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input,
    )
    # this is 500 because we explicitly pass in `throw` as True, which
    # raises an internal error in the module
    assert response.status_code == 500


def test_inference_sample_task_multipart_input(sample_task_model_id, client):
    """Simple check that we can submit multipart requests"""

    multipart_input = {
        "model_id": sample_task_model_id,
        "inputs.name": "world",
        "parameters": json.dumps({"throw": False}),
    }

    response = client.post(f"/api/v1/task/sample", files=multipart_input)

    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["greeting"] == "Hello world"

    multipart_input["parameters"] = json.dumps({"throw": True})
    response = client.post(
        f"/api/v1/task/sample",
        files=multipart_input,
    )
    # this is 500 because we explicitly pass in `throw` as True, which
    # raises an internal error in the module
    assert response.status_code == 500


def test_inference_file_task_multipart_flipped_input(file_task_model_id, client):
    """Ensure that multiple multipart json inputs are merged together instead of overriding"""
    # cGRmZGF0Yf//AA== is b"pdfdata\xff\xff\x00" base64 encoded
    temp_file = tempfile.NamedTemporaryFile()
    temp_file_name = Path(temp_file.name).name
    temp_file.write(b"pdfdata\xff\xff\x00")
    temp_file.flush()
    temp_file.seek(0)

    file_input = {
        "model_id": file_task_model_id,
        "inputs.file": temp_file,
        "inputs": json.dumps({"metadata": {"name": "agoodname"}}),
    }

    response = client.post(
        f"/api/v1/task/file",
        files=file_input,
    )
    content_stream = BytesIO(response.content)

    assert response.status_code == 200
    with zipfile.ZipFile(content_stream) as output_file:
        assert len(output_file.namelist()) == 2
        assert "metadata.json" in output_file.namelist()
        assert f"processed_{temp_file_name}" in output_file.namelist()

        with output_file.open(f"processed_{temp_file_name}") as pdf_result:
            assert pdf_result.read() == b"bounding|pdfdata\xff\xff\x00|box"


def test_inference_other_task(other_task_model_id, client):
    """Simple check that we can ping a model"""
    json_input = {"model_id": other_task_model_id, "inputs": {"name": "world"}}
    response = client.post(
        f"/api/v1/task/other",
        json=json_input,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["farewell"] == "goodbye: world 42 times"


def test_output_file_task(file_task_model_id, client):
    """Simple check that we can get a file output"""
    # cGRmZGF0Yf//AA== is b"pdfdata\xff\xff\x00" base64 encoded
    temp_file = tempfile.NamedTemporaryFile()
    temp_file_name = Path(temp_file.name).name
    temp_file.write(b"pdfdata\xff\xff\x00")
    temp_file.flush()
    temp_file.seek(0)

    file_input = {
        "model_id": file_task_model_id,
        "inputs.file": temp_file,
        "inputs.metadata": json.dumps({"name": "agoodname"}),
    }

    response = client.post(
        f"/api/v1/task/file",
        files=file_input,
    )
    content_stream = BytesIO(response.content)

    assert response.status_code == 200
    with zipfile.ZipFile(content_stream) as output_file:
        assert len(output_file.namelist()) == 2
        assert "metadata.json" in output_file.namelist()
        assert f"processed_{temp_file_name}" in output_file.namelist()

        with output_file.open(f"processed_{temp_file_name}") as pdf_result:
            assert pdf_result.read() == b"bounding|pdfdata\xff\xff\x00|box"


def test_invalid_input_exception(file_task_model_id, client):
    """Simple check that the server catches caikit core exceptions"""
    json_file_input = {
        "model_id": file_task_model_id,
        "inputs": {
            "file": {
                # cGRmZGF0Yf//AA== is b"pdfdata\xff\xff\x00" base64 encoded
                "data": "cGRmZGF0Yf//AA==",
                "filename": "unsupported_file.exe",
            },
            "metadata": {
                "name": "agoodname",
            },
        },
    }

    response = client.post(
        f"/api/v1/task/file",
        json=json_file_input,
    )
    assert response.status_code == 400
    json_response = response.json()
    assert json_response["details"] == "Executables are not a supported File type"


@pytest.mark.skip(
    "Skipping testing streaming cases with FastAPI's testclient, pending resolution https://github.com/tiangolo/fastapi/discussions/10518"
)
def test_inference_streaming_sample_module(sample_task_model_id, client):
    """Simple check for testing a happy path unary-stream case"""
    json_input = {"model_id": sample_task_model_id, "inputs": {"name": "world"}}
    # send in multiple requests just to check
    for i in range(10):
        stream = client.post(
            f"/api/v1/task/server-streaming-sample",
            json=json_input,
        )
        assert stream.status_code == 200
        stream_content = stream.content.decode(stream.default_encoding)
        stream_responses = json.loads(
            "[{}]".format(
                stream_content.replace("event: ", '{"event":')
                .replace(
                    StreamEventTypes.MESSAGE.value,
                    '"' + f"{StreamEventTypes.MESSAGE.value}" + '"}',
                )
                .replace("data: ", "")
                .replace("\r\n", "")
                .replace("}{", "}, {")
            )
        )
        assert len(stream_responses) == 20
        assert all(
            resp.get("greeting") == "Hello world stream"
            for resp in stream_responses
            if "greeting" in resp
        )
        assert all(
            resp.get("event") == StreamEventTypes.MESSAGE.value
            for resp in stream_responses
            if "event" in resp
        )


def test_inference_streaming_sample_module_actual_server(
    sample_task_model_id, runtime_http_server
):
    """Simple check for testing a happy path unary-stream case
    but pings the actual running server"""

    for i in range(10):
        input = {"model_id": sample_task_model_id, "inputs": {"name": f"world{i}"}}
        url = f"http://localhost:{runtime_http_server.port}/api/v1/task/server-streaming-sample"
        stream = requests.post(url=url, json=input, verify=False)
        assert stream.status_code == 200
        stream_content = stream.content.decode(stream.encoding)
        stream_responses = json.loads(
            "[{}]".format(
                stream_content.replace("event: ", '{"event":')
                .replace(
                    StreamEventTypes.MESSAGE.value,
                    '"' + f"{StreamEventTypes.MESSAGE.value}" + '"}',
                )
                .replace("data: ", "")
                .replace("\r\n", "")
                .replace("}{", "}, {")
            )
        )
        assert len(stream_responses) == 20
        assert all(
            resp.get("greeting") == f"Hello world{i} stream"
            for resp in stream_responses
            if "greeting" in resp
        )
        assert all(
            resp.get("event") == StreamEventTypes.MESSAGE.value
            for resp in stream_responses
            if "event" in resp
        )


def test_inference_streaming_sample_module_actual_server_throws(
    sample_task_model_id, runtime_http_server
):
    """Simple check for testing an exception in unary-stream case
    that pings the actual running server"""

    for i in range(10):
        input = {
            "model_id": sample_task_model_id,
            "inputs": {"name": f"world{i}"},
            "parameters": {"err_stream": True},
        }
        url = f"http://localhost:{runtime_http_server.port}/api/v1/task/server-streaming-sample"
        stream = requests.post(url=url, json=input, verify=False)
        assert stream.status_code == 200
        stream_content = stream.content.decode(stream.encoding)
        stream_responses = json.loads(
            "[{}]".format(
                stream_content.replace("event: ", '{"event":')
                .replace(
                    StreamEventTypes.ERROR.value,
                    '"' + f"{StreamEventTypes.ERROR.value}" + '"}',
                )
                .replace("data: ", "")
                .replace("\r\n", "")
                .replace("}{", "}, {")
            )
        )
        assert len(stream_responses) == 2
        assert stream_responses[0].get("event") == StreamEventTypes.ERROR.value
        assert (
            stream_responses[1].get("details") == "ValueError('raising a ValueError')"
        )
        assert stream_responses[1].get("code") == 400


def test_inference_malformed_param(client):
    """Send a malformed data parameter field to the inference call to induce the correct HTTP error"""

    response = client.post(
        "/api/v1/task/sample",
        data='{"bad_input": 100,}',  # send intentionally bad json
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422

    json_response = response.json()

    assert "Invalid JSON" in json_response["details"]
    assert json_response["additional_info"][0]["type"] == "json_invalid"


def test_inference_non_serializable_json(client):
    """Send non_serializable json as the data parameter field to the inference call to test correct error handling"""

    byte_data = bytes([1, 2, 3, 4, 5])
    base64_data = base64.b64encode(byte_data)

    response = client.post(
        "/api/v1/task/sample",
        data=base64_data,  # send byte object
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422

    json_response = response.json()

    assert "Invalid JSON" in json_response["details"]
    assert json_response["additional_info"][0]["type"] == "json_invalid"


def test_no_model_id(client):
    """Simple check to make sure we return a 400 if no model_id in payload"""
    response = client.post(
        f"/api/v1/task/sample",
        json={"inputs": {"name": "world"}},
    )
    assert response.status_code == 400
    assert "Please provide model_id in payload" in response.json()["details"]


def test_inference_multi_task_module(multi_task_model_id, client):
    """Simple check that we can ping a model"""
    # cGRmZGF0Yf//AA== is b"pdfdata\xff\xff\x00" base64 encoded
    json_input = {
        "model_id": multi_task_model_id,
        "inputs": {"filename": "example.pdf", "data": "cGRmZGF0Yf//AA=="},
    }
    response = client.post(
        f"/api/v1/task/second",
        json=json_input,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["farewell"] == "Goodbye from SecondTask"


def test_model_not_found(client):
    """Simple error check to make sure we return a 404 in case of
    incorrect model_id"""
    response = client.post(
        f"/api/v1/task/sample",
        json={"model_id": "not_an_id", "inputs": {"name": "world"}},
    )
    assert response.status_code == 404


def test_model_not_found_with_lazy_load_multi_model_finder(open_port):
    """An error check to make sure we return a 404 in case of
    incorrect model_id while using multi model finder with lazy load enabled"""
    with tempfile.TemporaryDirectory() as workdir:
        # NOTE: This test requires that the ModelManager class not be a singleton.
        #   To accomplish this, the singleton instance is temporarily removed.
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": workdir,
                    "lazy_load_local_models": True,
                },
                "model_management": {
                    "finders": {
                        "default": {
                            "type": MultiModelFinder.name,
                            "config": {
                                "finder_priority": ["local"],
                            },
                        },
                        "local": {"type": "LOCAL"},
                    }
                },
            },
            "merge",
        ):
            with runtime_http_test_server(open_port) as server:
                # double checking that our local model_management change took affect
                assert (
                    server.global_predict_servicer._model_manager._lazy_load_local_models
                )
                response = requests.post(
                    f"http://localhost:{server.port}/api/v1/task/sample",
                    json={"model_id": "not_an_id", "inputs": {"name": "world"}},
                )
                assert response.status_code == 404


def test_inference_sample_task_incorrect_input(sample_task_model_id, client):
    """Test that with an incorrect input, we get back a 422"""
    json_input = {
        "model_id": sample_task_model_id,
        "inputs": {"blah": "world"},
    }
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input,
    )
    assert response.status_code == 422
    json_response = response.json()
    # assert standard fields in the response
    assert json_response["details"] is not None
    assert json_response["code"] is not None
    assert json_response["id"] is not None
    assert json_response["details"] == "Extra inputs are not permitted"


@pytest.mark.skip("Skipping since we're not tacking forward compatibility atm")
def test_inference_sample_task_forward_compatibility(sample_task_model_id, client):
    """Test that clients can send in params that don't exist on server
    without any error"""
    json_input = {
        "model_id": sample_task_model_id,
        "inputs": {"name": "world", "blah": "blah"},
    }
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["greeting"] == "Hello world"


def test_health_check_ok(client):
    """Make sure the health check returns OK"""
    response = client.get(http_server.HEALTH_ENDPOINT)
    assert response.status_code == 200
    assert response.text == "OK"


## Info Tests ##################################################################


def test_runtime_info_ok(runtime_http_server):
    """Make sure the runtime info returns version data"""
    with TestClient(runtime_http_server.app) as client:
        response = client.get(http_server.RUNTIME_INFO_ENDPOINT)
        assert response.status_code == 200

        json_response = response.json()
        assert "caikit" in json_response["python_packages"]
        # runtime_version not added if not set
        assert json_response["runtime_version"] == ""
        # dependent libraries not added if all packages not set to true
        assert "py_to_proto" not in json_response["python_packages"]


def test_runtime_info_ok_response_all_packages(runtime_http_server):
    with temp_config(
        {
            "runtime": {
                "version_info": {
                    "python_packages": {
                        "all": True,
                    },
                    "runtime_image": "1.2.3",
                }
            },
        },
        "merge",
    ):
        with TestClient(runtime_http_server.app) as client:
            response = client.get(http_server.RUNTIME_INFO_ENDPOINT)
            assert response.status_code == 200

            json_response = response.json()
            assert json_response["runtime_version"] == "1.2.3"
            assert "caikit" in json_response["python_packages"]
            # dependent libraries versions added
            assert "alog" in json_response["python_packages"]
            assert "py_to_proto" in json_response["python_packages"]


def test_runtime_info_ok_custom_python_packages(runtime_http_server):
    """Make sure the runtime info returns version data"""
    with temp_config(
        {"runtime": {"version_info": {"python_packages": {"custom_package": "0.1.0"}}}},
        merge_strategy="merge",
    ):
        with TestClient(runtime_http_server.app) as client:
            response = client.get(http_server.RUNTIME_INFO_ENDPOINT)
            assert response.status_code == 200

            json_response = response.json()
            # runtime_version not added if not set
            assert json_response["runtime_version"] == ""
            # custom library is set while other random packages are not
            assert "caikit" in json_response["python_packages"]
            assert json_response["python_packages"]["custom_package"] == "0.1.0"
            assert "py_to_proto" not in json_response["python_packages"]


def test_all_models_info_ok(client, sample_task_model_id):
    """Make sure the runtime info returns version data"""
    response = client.get(http_server.MODELS_INFO_ENDPOINT)
    assert response.status_code == 200

    json_response = response.json()
    # Assert some models are loaded
    assert len(json_response["models"]) > 0

    found_sample_task = False
    for model in json_response["models"]:
        # Assert name and id exist
        assert model["name"] and model["module_id"]
        if model["name"] == sample_task_model_id:
            assert model["module_metadata"]["name"] == "SampleModule"
            found_sample_task = True

    assert found_sample_task, "Unable to find sample_task model in models list"


def test_single_models_info_ok(client, sample_task_model_id):
    """Make sure the runtime info returns version data"""
    response = client.get(
        http_server.MODELS_INFO_ENDPOINT, params={"model_ids": sample_task_model_id}
    )
    assert response.status_code == 200

    json_response = response.json()
    # Assert some models are loaded
    assert len(json_response["models"]) == 1

    model = json_response["models"][0]
    assert model["name"] == sample_task_model_id
    assert model["module_metadata"]["name"] == "SampleModule"


## Train Tests #################################################################


def test_train_sample_task(client, runtime_http_server):
    model_name = "sample_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "training_data": {"data_stream": {"data": [{"number": 1}]}},
            "batch_size": 42,
        },
    }
    training_response = client.post(
        "/api/v1/SampleTaskSampleModuleTrain",
        json=json_input,
    )

    # assert training response
    training_json_response = training_response.json()
    assert training_response.status_code == 200, training_json_response
    assert (training_id := training_json_response["training_id"])
    assert training_json_response["model_name"] == model_name

    # assert trained model
    result = MODEL_MANAGER.get_model_future(training_id).load()
    assert result.batch_size == 42
    assert (
        result.MODULE_CLASS
        == "sample_lib.modules.sample_task.sample_implementation.SampleModule"
    )

    # register the newly trained model for inferencing
    register_trained_model(
        runtime_http_server.global_predict_servicer,
        model_name,
        training_id,
    )

    # test inferencing on new model
    json_input_inference = {"model_id": model_name, "inputs": {"name": "world"}}
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input_inference,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["greeting"] == "Hello world"


def test_train_sample_task_throws_s3_value_error(client):
    """test that if we provide s3 path, it throws an error"""
    model_name = "sample_task_train"
    json_input = {
        "model_name": model_name,
        "output_path": {"path": "non-existent path_to_s3"},
        "parameters": {
            "training_data": {"data_stream": {"data": [{"number": 1}]}},
            "batch_size": 42,
        },
    }
    training_response = client.post(
        f"/api/v1/SampleTaskSampleModuleTrain",
        json=json_input,
    )
    assert (
        "S3 output path not supported by this runtime"
        in training_response.json()["details"]
    )
    assert training_response.status_code == 500, training_response.json()


def test_train_primitive_task(client, runtime_http_server):
    model_name = "primitive_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "sample_input": {"name": "test"},
            "simple_list": ["hello", "world"],
            "union_list": ["hello", "world"],
            "union_list2": ["hello", "world"],
            "union_list3": ["hello", "world"],
            "union_list4": 1,
            "training_params_json_dict_list": [{"foo": {"bar": [1, 2, 3]}}],
            "training_params_json_dict": {"foo": {"bar": [1, 2, 3]}},
            "training_params_dict": {"layer_sizes": 100, "window_scaling": 200},
            "training_params_dict_int": {1: 0.1, 2: 0.01},
        },
    }

    training_response = client.post(
        f"/api/v1/SampleTaskSamplePrimitiveModuleTrain",
        json=json_input,
    )
    # assert training response
    training_json_response = training_response.json()
    assert training_response.status_code == 200, training_json_response
    assert (training_id := training_json_response["training_id"])
    assert training_json_response["model_name"] == model_name

    # assert trained model
    result = MODEL_MANAGER.get_model_future(training_id).load()
    assert result.training_params_dict == {
        "layer_sizes": 100,
        "window_scaling": 200,
    }
    assert result.training_params_json_dict == {"foo": {"bar": [1, 2, 3]}}
    assert (
        result.MODULE_CLASS
        == "sample_lib.modules.sample_task.primitive_party_implementation.SamplePrimitiveModule"
    )

    # register the newly trained model for inferencing
    register_trained_model(
        runtime_http_server.global_predict_servicer,
        model_name,
        training_id,
    )

    # test inferencing on new model
    json_input_inference = {"model_id": model_name, "inputs": {"name": "world"}}
    response = client.post(
        f"/api/v1/task/sample",
        json=json_input_inference,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["greeting"] == "hello: primitives! [1, 2, 3] 100"


def test_train_other_task(client, runtime_http_server):
    model_name = "other_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "training_data": {"data_stream": {"data": [1, 2]}},
            "sample_input": {"name": "test"},
        },
    }

    training_response = client.post(
        f"/api/v1/OtherTaskOtherModuleTrain",
        json=json_input,
    )
    # assert training response
    training_json_response = training_response.json()
    assert training_response.status_code == 200, training_json_response
    assert (training_id := training_json_response["training_id"])
    assert training_json_response["model_name"] == model_name

    # assert trained model
    result = MODEL_MANAGER.get_model_future(training_id).load()
    assert result.batch_size == 64
    assert (
        result.MODULE_CLASS
        == "sample_lib.modules.other_task.other_implementation.OtherModule"
    )

    # register the newly trained model for inferencing
    register_trained_model(
        runtime_http_server.global_predict_servicer,
        model_name,
        training_id,
    )

    # test inferencing on new model
    json_input_inference = {"model_id": model_name, "inputs": {"name": "world"}}
    response = client.post(
        f"/api/v1/task/other",
        json=json_input_inference,
    )
    json_response = response.json()
    assert response.status_code == 200, json_response
    assert json_response["farewell"] == "goodbye: world 64 times"


def test_train_long_running_sample_task(client, runtime_http_server):
    """Test that with a long running training job, the request returns before the training completes"""
    model_name = "sample_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "training_data": {"data_stream": {"data": [{"number": 1}]}},
            "batch_size": 42,
            "sleep_time": 5,  # mimic long train time
        },
    }
    training_response = client.post(
        f"/api/v1/SampleTaskSampleModuleTrain",
        json=json_input,
    )

    # assert training response received before training completed
    training_json_response = training_response.json()
    assert training_response.status_code == 200, training_json_response
    assert (training_id := training_json_response["training_id"])
    assert training_json_response["model_name"] == model_name

    # assert that the training is still running
    model_future = MODEL_MANAGER.get_model_future(training_id)
    assert model_future.get_info().status == TrainingStatus.RUNNING

    # Cancel the training
    model_future.cancel()
    assert model_future.get_info().status == TrainingStatus.CANCELED
    assert model_future.get_info().status.is_terminal


## Management Tests ############################################################


def test_model_management_deploy_lifecycle(open_port, deploy_good_model_files):
    """Test that models can be deployed/undeployed and reflect in the
    local_models_dir
    """
    with tempfile.TemporaryDirectory() as workdir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": workdir,
                    "lazy_load_local_models": True,
                },
            },
            "merge",
        ):
            with runtime_http_test_server(open_port) as server:
                with client_context(server) as client:

                    # Make sure no models loaded initially
                    resp = client.get(http_server.MODELS_INFO_ENDPOINT)
                    resp.raise_for_status()
                    model_info = resp.json()
                    assert len(model_info["models"]) == 0

                    # Do the deploy
                    model_id = "my-model"
                    deploy_req = {
                        "model_id": model_id,
                        "model_files": [
                            {
                                "filename": fname,
                                "data": base64.b64encode(data).decode("utf-8"),
                            }
                            for fname, data in deploy_good_model_files.items()
                        ],
                    }
                    resp = client.post(
                        http_server.MODEL_MANAGEMENT_ENDPOINT, json=deploy_req
                    )
                    resp.raise_for_status()
                    resp_json = resp.json()
                    assert resp_json["name"] == model_id
                    model_path = os.path.join(workdir, model_id)
                    assert resp_json["model_path"] == model_path
                    assert os.path.isdir(model_path)

                    # Make sure the model shows up in info
                    resp = client.get(http_server.MODELS_INFO_ENDPOINT)
                    resp.raise_for_status()
                    model_info = resp.json()
                    assert len(model_info["models"]) == 1
                    assert model_info["models"][0]["name"] == model_id

                    # Make sure an appropriate error is raised for trying to
                    # deploy the same model again
                    resp = client.post(
                        http_server.MODEL_MANAGEMENT_ENDPOINT, json=deploy_req
                    )
                    assert resp.status_code == 409

                    # Undeploy the model
                    resp = client.delete(
                        http_server.MODEL_MANAGEMENT_ENDPOINT,
                        params={"model_id": model_id},
                    )
                    resp.raise_for_status()

                    # Make sure no models loaded
                    assert not client.get(http_server.MODELS_INFO_ENDPOINT).json()[
                        "models"
                    ]

                    # Make sure a 404 is raised if undeployed again
                    resp = client.delete(
                        http_server.MODEL_MANAGEMENT_ENDPOINT,
                        params={"model_id": model_id},
                    )
                    assert resp.status_code == 404


def test_model_management_deploy_invalid(open_port):
    """Test that attempting to deploy an invalid model fails"""
    with tempfile.TemporaryDirectory() as workdir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": workdir,
                    "lazy_load_local_models": True,
                },
            },
            "merge",
        ):
            with runtime_http_test_server(open_port) as server:
                with client_context(server) as client:

                    # Make sure no models loaded initially
                    resp = client.get(http_server.MODELS_INFO_ENDPOINT)
                    resp.raise_for_status()
                    model_info = resp.json()
                    assert len(model_info["models"]) == 0

                    # Do the deploy
                    model_id = "my-model"
                    deploy_req = {
                        "model_id": model_id,
                        "model_files": [
                            {
                                "filename": "foo.txt",
                                "data": "yikes",  # <- not base64
                            }
                        ],
                    }
                    resp = client.post(
                        http_server.MODEL_MANAGEMENT_ENDPOINT, json=deploy_req
                    )
                    assert resp.status_code == 422  # Unprocessable


def test_training_management_get_status(client):
    """Test that training status can be retrieved"""
    model_name = "sample_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "training_data": {"data_stream": {"data": [{"number": 1}]}},
            "batch_size": 42,
        },
    }
    training_response = client.post(
        "/api/v1/SampleTaskSampleModuleTrain",
        json=json_input,
    )

    # Start the training and make sure it starts successfully
    training_json_response = training_response.json()
    training_response.raise_for_status()

    # Get the status for the training
    training_id = training_json_response["training_id"]
    get_response = client.get(
        http_server.TRAINING_MANAGEMENT_ENDPOINT, params={"training_id": training_id}
    )
    get_response.raise_for_status()


def test_training_management_cancel(client):
    """Test that trainings can be canceled"""
    model_name = "sample_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "training_data": {"data_stream": {"data": [{"number": 1}]}},
            "batch_size": 42,
            # Sleep the job so that cancellation can interrupt it. It will not
            # sleep for this long in practice.
            "sleep_time": 20,
        },
    }
    training_response = client.post(
        "/api/v1/SampleTaskSampleModuleTrain",
        json=json_input,
    )

    # Start the training and make sure it starts successfully
    training_json_response = training_response.json()
    training_response.raise_for_status()

    # Cancel the training
    training_id = training_json_response["training_id"]
    cancel_response = client.delete(
        http_server.TRAINING_MANAGEMENT_ENDPOINT, params={"training_id": training_id}
    )
    cancel_response.raise_for_status()

    # Make sure the status reflects being canceled
    get_response = client.get(
        http_server.TRAINING_MANAGEMENT_ENDPOINT, params={"training_id": training_id}
    )
    get_response.raise_for_status()
    assert get_response.json()["state"] == "CANCELED"


def test_training_management_errors(client):
    """Test that training status can be retrieved"""
    model_name = "sample_task_train"
    json_input = {
        "model_name": model_name,
        "parameters": {
            "training_data": {"data_stream": {"data": [{"number": 1}]}},
            "batch_size": 42,
        },
    }
    training_response = client.post(
        "/api/v1/SampleTaskSampleModuleTrain",
        json=json_input,
    )

    # Make sure unknown training GET returns 404
    assert (
        client.get(
            http_server.TRAINING_MANAGEMENT_ENDPOINT, params={"training_id": "bad-id"}
        ).status_code
        == 404
    )

    # Make sure missing param returns 422
    assert client.get(http_server.TRAINING_MANAGEMENT_ENDPOINT).status_code == 422

    # Make sure unknown training DELETE returns 404
    assert (
        client.delete(
            http_server.TRAINING_MANAGEMENT_ENDPOINT, params={"training_id": "bad-id"}
        ).status_code
        == 404
    )
