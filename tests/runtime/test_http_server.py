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
from typing import Dict, List
import json
import os
import signal
import tempfile

# Third Party
from fastapi.testclient import TestClient
import pytest
import requests
import tls_test_tools

# Local
from caikit.core import MODEL_MANAGER, DataObjectBase, dataobject
from caikit.core.data_model.base import DataBase
from caikit.interfaces.nlp.data_model import GeneratedTextStreamResult, GeneratedToken
from caikit.runtime import http_server
from tests.conftest import temp_config
from tests.runtime.conftest import (
    ModuleSubproc,
    register_trained_model,
    runtime_http_test_server,
)

## Helpers #####################################################################


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


@dataclass
class KeyPair:
    cert: str
    key: str


@dataclass
class TLSConfig:
    server: KeyPair
    client: KeyPair


@contextmanager
def generate_tls_configs(
    port: int, tls: bool = False, mtls: bool = False, **http_config_overrides
) -> Dict[str, Dict]:
    """Helper to generate tls configs"""
    with tempfile.TemporaryDirectory() as workdir:
        config_overrides = {}
        client_keyfile, client_certfile, ca_certfile = None, None, None
        if mtls or tls:
            ca_key = tls_test_tools.generate_key()[0]
            ca_cert = tls_test_tools.generate_ca_cert(ca_key)
            ca_certfile, _ = save_key_cert_pair("ca", workdir, cert=ca_cert)
            server_certfile, server_keyfile = save_key_cert_pair(
                "server",
                workdir,
                *tls_test_tools.generate_derived_key_cert_pair(ca_key=ca_key),
            )

            tls_config = TLSConfig(
                server=KeyPair(cert=server_certfile, key=server_keyfile),
                client=KeyPair(cert="", key=""),
            )
            # need to save this ca_certfile in config_overrides so the tls tests below can access it from client side
            config_overrides["use_in_test"] = {"ca_cert": ca_certfile}

            # also saving a bad ca_certfile for a failure test case
            bad_ca_file = os.path.join(workdir, "bad_ca_cert.crt")
            with open(bad_ca_file, "w") as handle:
                bad_cert = (
                    "-----BEGIN CERTIFICATE-----\nfoobar\n-----END CERTIFICATE-----"
                )
                handle.write(bad_cert)
            config_overrides["use_in_test"]["bad_ca_cert"] = bad_ca_file

            if mtls:
                client_certfile, client_keyfile = save_key_cert_pair(
                    "client",
                    workdir,
                    *tls_test_tools.generate_derived_key_cert_pair(ca_key=ca_key),
                )
                tls_config.client = KeyPair(cert=ca_certfile, key="")
                # need to save the client cert and key in config_overrides so the mtls test below can access it
                config_overrides["use_in_test"]["client_cert"] = client_certfile
                config_overrides["use_in_test"]["client_key"] = client_keyfile

            config_overrides["runtime"] = {"tls": tls_config}
        config_overrides.setdefault("runtime", {})["http"] = {
            "server_shutdown_grace_period_seconds": 0.01,  # this is so the server is killed after 0.1 if no test is running
            "port": port,
            **http_config_overrides,
        }

        with temp_config(config_overrides, "merge"):
            yield config_overrides


## Insecure and TLS Tests #######################################################################


def test_insecure_server(runtime_http_server, open_port):
    with generate_tls_configs(open_port):
        # start a non-blocking http server
        resp = requests.get(f"http://localhost:{runtime_http_server.port}/docs")
        resp.raise_for_status()


def test_basic_tls_server(open_port):
    with generate_tls_configs(
        open_port, tls=True, mtls=False, http_config_overrides={}
    ) as config_overrides:
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
    with generate_tls_configs(
        open_port, tls=True, mtls=False, http_config_overrides={}
    ) as config_overrides:
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
    with generate_tls_configs(
        open_port, tls=True, mtls=True, http_config_overrides={}
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
    with generate_tls_configs(
        open_port, tls=True, mtls=True, http_config_overrides={}
    ) as config_overrides:
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
    "enabled_services",
    [(True, False), (False, True), (False, False)],
)
def test_services_disabled(open_port, enabled_services):
    enable_inference, enable_training = enabled_services
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
            assert (server.global_predict_servicer and enable_inference) or (
                server.global_predict_servicer is None and not enable_inference
            )
            assert server.enable_training == enable_training
            # TODO: Update once training enabled
            # assert (server.global_train_servicer and enable_training) or (
            #     server.global_train_servicer is None and not enable_training
            # )


## Functional Tests #######################################################################


def test_build_dm_object_simple(runtime_http_server):
    """Test building our simple DM objects through pydantic objects"""

    # get our DM class
    str_sequence_dm_class = DataBase.get_class_for_name("StrSequence")
    # get pydantic model for our DM class
    str_sequence_pydantic_model = http_server.PYDANTIC_TO_DM_MAPPING.get(
        str_sequence_dm_class
    )
    # build our DM object using a pydantic object
    str_sequence_dm_obj = runtime_http_server._build_dm_object(
        str_sequence_pydantic_model(values=["one", "two"])
    )

    # assert it's our DM object, all fine and dandy
    assert isinstance(str_sequence_dm_obj, DataBase)
    assert str_sequence_dm_obj.to_json() == '{"values": ["one", "two"]}'


def test_build_dm_object_datastream(runtime_http_server):
    """Test building our datastream DM objects through pydantic objects"""

    # get our DM class
    datastream_dm_class = DataBase.get_class_for_name(
        "DataStreamSourceSampleTrainingType"
    )
    # get pydantic model for our DM class
    datastream_pydantic_model = http_server.PYDANTIC_TO_DM_MAPPING.get(
        datastream_dm_class
    )
    # build our DM object using a pydantic object
    datastream_dm_obj = runtime_http_server._build_dm_object(
        datastream_pydantic_model(data_stream={"data": [{"number": 1}, {"number": 2}]})
    )

    # assert it's our DM object, all fine and dandy
    assert isinstance(datastream_dm_obj, DataBase)
    assert (
        datastream_dm_obj.to_json()
        == '{"jsondata": {"data": [{"number": 1}, {"number": 2}]}}'
    )


def test_pydantic_wrapping_with_enums():
    """Check that the pydantic wrapping works on our data models when they have enums"""
    # The NLP GeneratedTextStreamResult data model contains enums

    # Check that our data model is fine and dandy
    token = GeneratedToken(text="foo")
    assert token.text == "foo"

    # Wrap the containing data model in pydantic
    http_server.RuntimeHTTPServer._dataobject_to_pydantic(GeneratedTextStreamResult)

    # Check that our data model is _still_ fine and dandy
    token = GeneratedToken(text="foo")
    assert token.text == "foo"


def test_pydantic_wrapping_with_lists(runtime_http_server):
    """Check that pydantic wrapping works on data models with lists"""

    @dataobject(package="http")
    class BarTest(DataObjectBase):
        baz: int

    @dataobject(package="http")
    class FooTest(DataObjectBase):
        bars: List[BarTest]

    foo = FooTest(bars=[BarTest(1)])
    assert foo.bars[0].baz == 1

    runtime_http_server._dataobject_to_pydantic(FooTest)

    foo = FooTest(bars=[BarTest(1)])
    assert foo.bars[0].baz == 1


## Inference Tests #######################################################################


def test_docs(runtime_http_server):
    """Simple check that pinging /docs returns 200"""
    with TestClient(runtime_http_server.app) as client:
        response = client.get("/docs")
        assert response.status_code == 200


def test_docs_using_running_http_server(runtime_http_server):
    """Simple check that pinging /docs returns 200
    but pints the actual running server"""
    response = requests.get(f"http://localhost:{runtime_http_server.port}/docs")
    assert response.status_code == 200


def test_inference_sample_task(sample_task_model_id, runtime_http_server):
    """Simple check that we can ping a model"""
    with TestClient(runtime_http_server.app) as client:
        json_input = {"inputs": {"name": "world"}}
        response = client.post(
            f"/api/v1/{sample_task_model_id}/task/sample",
            json=json_input,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["greeting"] == "Hello world"


def test_inference_sample_task_optional_field(
    sample_task_model_id, runtime_http_server
):
    """Simple check for optional fields"""
    with TestClient(runtime_http_server.app) as client:
        json_input = {
            "inputs": {"name": "world"},
            "parameters": {"throw": True},
        }
        response = client.post(
            f"/api/v1/{sample_task_model_id}/task/sample",
            json=json_input,
        )
        # this is 500 because we explicitly pass in `throw` as True, which
        # raises an internal error in the module
        assert response.status_code == 500


def test_inference_other_task(other_task_model_id, runtime_http_server):
    """Simple check that we can ping a model"""
    with TestClient(runtime_http_server.app) as client:
        json_input = {"inputs": {"name": "world"}}
        response = client.post(
            f"/api/v1/{other_task_model_id}/task/other",
            json=json_input,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["farewell"] == "goodbye: world 42 times"


def test_inference_streaming_sample_module(sample_task_model_id, runtime_http_server):
    """Simple check for testing a happy path unary-stream case"""
    with TestClient(runtime_http_server.app) as client:
        json_input = {"inputs": {"name": "world"}}
        stream = client.post(
            f"/api/v1/{sample_task_model_id}/task/server-streaming-sample",
            json=json_input,
        )
        assert stream.status_code == 200
        assert (
            stream.content.decode(stream.default_encoding).count("SampleOutputType")
            == 10
        )
        assert (
            b"SampleOutputType(greeting='Hello world stream')\r\n\r\n" in stream.content
        )


def test_model_not_found(runtime_http_server):
    """Simple check that we can ping a model"""
    with TestClient(runtime_http_server.app) as client:
        response = client.post(
            f"/api/v1/this_is_not_a_model/task/sample",
            json={"inputs": {"name": "world"}},
        )
        assert response.status_code == 404


def test_inference_sample_task_throws_incorrect_input(
    sample_task_model_id, runtime_http_server
):
    """error check for a request with incorrect input"""
    with TestClient(runtime_http_server.app) as client:
        json_input = {"blah": {"sample_input": {"name": "world"}}}
        response = client.post(
            f"/api/v1/{sample_task_model_id}/task/sample",
            json=json_input,
        )
        assert response.status_code == 400


def test_health_check_ok(runtime_http_server):
    """Make sure the health check returns OK"""
    with TestClient(runtime_http_server.app) as client:
        response = client.get(http_server.HEALTH_ENDPOINT)
        assert response.status_code == 200
        assert response.text == "OK"


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


## Train Tests #######################################################################


def test_train_sample_task(runtime_http_server):
    model_name = "sample_task_train"
    with TestClient(runtime_http_server.app) as client:
        json_input = {
            "inputs": {
                "model_name": model_name,
                "training_data": {"data_stream": {"data": [{"number": 1}]}},
                # "training_data": {"data_stream": {"file": {"filename": "file1"}}},
            },
            "parameters": {"batch_size": 42},
        }
        training_response = client.post(
            f"/api/v1/SampleTaskSampleModuleTrain",
            json=json_input,
        )

        # assert training response
        assert training_response.status_code == 200
        training_json_response = json.loads(
            training_response.content.decode(training_response.default_encoding)
        )
        assert (training_id := training_json_response["training_id"])
        assert (model_name := training_json_response["model_name"]) == model_name

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
        json_input_inference = {"inputs": {"name": "world"}}
        response = client.post(
            f"/api/v1/{model_name}/task/sample",
            json=json_input_inference,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["greeting"] == "Hello world"


def test_train_sample_task_throws_s3_value_error(runtime_http_server):
    """test that if we provide s3 path, it throws an error"""
    model_name = "sample_task_train"
    with TestClient(runtime_http_server.app) as client:
        json_input = {
            "inputs": {
                "model_name": model_name,
                "training_data": {"data_stream": {"file": "hello"}},
                "output_path": {"path": "non-existent path_to_s3"},
            },
            "parameters": {"batch_size": 42},
        }
        training_response = client.post(
            f"/api/v1/SampleTaskSampleModuleTrain",
            json=json_input,
        )
        assert (
            "S3 output path not supported by this runtime"
            in training_response.content.decode(training_response.default_encoding)
        )
        assert training_response.status_code == 500


def test_train_primitive_task(runtime_http_server):
    model_name = "primitive_task_train"
    with TestClient(runtime_http_server.app) as client:
        json_input = {
            "inputs": {
                "model_name": model_name,
                "sample_input": {"name": "test"},
                "simple_list": ["hello", "world"],
                "union_list": ["hello", "world"],
                "union_list2": ["hello", "world"],
                "union_list3": ["hello", "world"],
                "union_list4": 1,
                "training_params_json_dict_list": [{"foo": {"bar": [1, 2, 3]}}],
            },
            "parameters": {
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
        assert training_response.status_code == 200
        training_json_response = json.loads(
            training_response.content.decode(training_response.default_encoding)
        )
        assert (training_id := training_json_response["training_id"])
        assert (model_name := training_json_response["model_name"]) == model_name

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
        json_input_inference = {"inputs": {"name": "world"}}
        response = client.post(
            f"/api/v1/{model_name}/task/sample",
            json=json_input_inference,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["greeting"] == "hello: primitives! [1, 2, 3] 100"


def test_train_other_task(runtime_http_server):
    model_name = "other_task_train"
    with TestClient(runtime_http_server.app) as client:
        json_input = {
            "inputs": {
                "model_name": model_name,
                "training_data": {"data_stream": {"data": [1, 2]}},
                "sample_input": {"name": "test"},
            }
        }

        training_response = client.post(
            f"/api/v1/OtherTaskOtherModuleTrain",
            json=json_input,
        )
        # assert training response
        assert training_response.status_code == 200
        training_json_response = json.loads(
            training_response.content.decode(training_response.default_encoding)
        )
        assert (training_id := training_json_response["training_id"])
        assert (model_name := training_json_response["model_name"]) == model_name

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
        json_input_inference = {"inputs": {"name": "world"}}
        response = client.post(
            f"/api/v1/{model_name}/task/other",
            json=json_input_inference,
        )
        assert response.status_code == 200
        json_response = json.loads(response.content.decode(response.default_encoding))
        assert json_response["farewell"] == "goodbye: world 64 times"
