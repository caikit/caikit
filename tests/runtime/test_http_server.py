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
from dataclasses import dataclass
from typing import Optional
import json
import os
import tempfile
import time

# Third Party
import pytest

# import requests
import tls_test_tools

# First Party
import aconfig

# Local
from caikit.runtime import http_server
from tests.conftest import temp_config

## Helpers #####################################################################


def save_key_cert_pair(prefix, workdir, key=None, cert=None):
    crtfile, keyfile = None, None
    if key is not None:
        keyfile = os.path.join(workdir, f"{prefix}.key")
        with open(keyfile, "w") as handle:
            handle.write(key)
    if cert is not None:
        crtfile = os.path.join(workdir, f"{prefix}.crt")
        with open(keyfile, "w") as handle:
            handle.write(cert)
    return crtfile, keyfile


@dataclass
class SampleServer:
    server: http_server.RuntimeHTTPServer
    port: int
    ca_certfile: Optional[str]
    client_keyfile: Optional[str]
    client_certfile: Optional[str]


@contextmanager
def sample_http_server(tls: bool = False, mtls: bool = False, **http_config_overrides):
    """Helper to boot up an instance of the http server on an available port"""
    with tempfile.TemporaryDirectory() as workdir:
        config_overrides = {}
        client_keyfile, client_certfile, ca_certfile = None, None, None
        if mtls or tls:
            ca_key = tls_test_tools.generate_key()[0]
            ca_certfile, _ = save_key_cert_pair(
                "ca", workdir, cert=tls_test_tools.generate_ca_cert(ca_key)
            )
            server_keyfile, server_certfile = save_key_cert_pair(
                "server",
                workdir,
                *tls_test_tools.generate_derived_key_cert_pair(ca_key),
            )
            config_overrides["tls"] = {
                "server": {
                    "key": server_keyfile,
                    "cert": server_certfile,
                }
            }
            if mtls:
                client_keyfile, client_certfile = save_key_cert_pair(
                    "client",
                    workdir,
                    *tls_test_tools.generate_derived_key_cert_pair(ca_key),
                )
                config_overrides["tls"]["client"] = {"cert": ca_certfile}
        port = http_server.RuntimeServerBase._find_port()
        config_overrides.setdefault("runtime", {})["http"] = {
            "port": port,
            **http_config_overrides,
        }
        with temp_config(config_overrides, "merge"):
            with http_server.RuntimeHTTPServer() as server:
                time.sleep(0.1)
                yield SampleServer(
                    server=server,
                    port=port,
                    ca_certfile=ca_certfile,
                    client_keyfile=client_keyfile,
                    client_certfile=client_certfile,
                )


@pytest.fixture
def insecure_http_server():
    with sample_http_server() as sample_server:
        yield sample_server


# Third Party
## Tests #######################################################################
from fastapi.testclient import TestClient

# def test_simple():
#     server = http_server.RuntimeHTTPServer()


def test_docs():
    """Simple check that pinging /docs returns 200"""
    server = http_server.RuntimeHTTPServer()
    with TestClient(server.app) as client:
        response = client.get("/docs")
        assert response.status_code == 200


def test_inference(sample_task_model_id):
    """Simple check that we can ping a model"""
    server = http_server.RuntimeHTTPServer()
    with TestClient(server.app) as client:
        json_input = {"inputs": {"sample_input": {"name": "world"}}}
        response = client.post(
            f"/api/v1/{sample_task_model_id}/task/sample",
            json=json_input,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["greeting"] == "Hello world"


def test_inference_optional_field(sample_task_model_id):
    """Simple check for optional fields"""
    server = http_server.RuntimeHTTPServer()
    with TestClient(server.app) as client:
        json_input = {
            "inputs": {"sample_input": {"name": "world"}},
            "parameters": {"throw": True},
        }
        response = client.post(
            f"/api/v1/{sample_task_model_id}/task/sample",
            json=json_input,
        )
        # this is 500 because we explicitly pass in `throw` as True, which
        # raises an internal error in the module
        assert response.status_code == 500


def test_inference_other_task(other_task_model_id):
    """Simple check that we can ping a model"""
    server = http_server.RuntimeHTTPServer()
    with TestClient(server.app) as client:
        json_input = {"inputs": {"sample_input": {"name": "world"}}}
        response = client.post(
            f"/api/v1/{other_task_model_id}/task/other",
            json=json_input,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["farewell"] == "goodbye: world 42 times"


def test_model_not_found():
    """Simple check that we can ping a model"""
    server = http_server.RuntimeHTTPServer()
    with TestClient(server.app) as client:
        response = client.post(
            f"/api/v1/this_is_not_a_model/task/sample",
            json={"inputs": {"name": "world"}},
        )
        assert response.status_code == 404


# TODO: uncomment later
# def test_train():
#     server = http_server.RuntimeHTTPServer()
#     with TestClient(server.app) as client:
#         json_input = {
#             "inputs": {
#                 "model_name": "sample_task_train",
#                 "training_data": {"jsondata": {"number": 1}},
#             }
#         }
#         response = client.post(
#             f"/api/v1/asdf/SampleTaskSampleModuleTrain",
#             json=json_input,
#         )
#         assert response.status_code == 200
#         json_response = json.loads(response.content.decode(response.default_encoding))
#         assert json_response["greeting"] == "Hello world"
