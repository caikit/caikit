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
Tests for the RemoteModelFinder
"""

# Standard
from contextlib import contextmanager
from typing import Optional
from unittest.mock import MagicMock, patch

# Third Party
import grpc
import pytest

# First Party
from aconfig import Config, ImmutableConfig

# Local
from caikit.interfaces.runtime.data_model import ModelInfo, ModelInfoResponse
from caikit.runtime.client import RemoteModelFinder, RemoteModuleConfig
from caikit.runtime.model_management.model_manager import ModelManager
from sample_lib.modules.file_processing import BoundingBoxModule
from sample_lib.modules.sample_task import SampleModule
from tests.conftest import random_test_id
from tests.fixtures import Fixtures
from tests.runtime.conftest import multi_task_model_id  # noqa: F401
from tests.runtime.conftest import sample_task_model_id  # noqa: F401
from tests.runtime.conftest import (  # noqa: F401
    generate_tls_configs,
    open_port,
    runtime_test_server,
)

## Test Helpers #######################################################################


@pytest.fixture
def sample_module_id(good_model_path) -> str:
    """Loaded model ID using model manager load model implementation"""
    model_id = random_test_id()
    model_manager = ModelManager.get_instance()
    # model load test already tests with archive - just using a model path here
    local_model = model_manager.load_model(
        model_id,
        local_model_path=good_model_path,
        model_type=Fixtures.get_good_model_type(),  # eventually we'd like to be determining the type from the model itself...
    )
    yield local_model.model().MODULE_ID


@contextmanager
def file_task_model_context(box_model_path, file_model_id=None) -> str:
    """Load file model id. This is copied from conftest except as
    a contextmanager"""
    model_id = file_model_id or random_test_id()
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


@contextmanager
def temp_finder(
    multi_finder_name="remote",
    multi_finder_cfg=None,
    connection_cfg=None,
    remote_connections_cfg=None,
    min_poll_time=0,
    protocol="grpc",
):
    # Provide defaults
    if not multi_finder_cfg:
        multi_finder_cfg = {
            "discover_models": True,
            "supported_models": {},
            "min_poll_time": min_poll_time,
        }

    if connection_cfg:
        multi_finder_cfg["connection"] = connection_cfg
    elif connection_cfg is None:
        multi_finder_cfg["connection"] = {
            "hostname": "localhost",
        }

    if remote_connections_cfg:
        multi_finder_cfg["remote_connections"] = remote_connections_cfg

    if "protocol" not in multi_finder_cfg:
        multi_finder_cfg["protocol"] = protocol

    yield RemoteModelFinder(ImmutableConfig(multi_finder_cfg), multi_finder_name)


## Tests #######################################################################


def test_remote_finder_static_model(sample_module_id):
    """Test to ensure static supported_models definition works as expected"""
    with temp_finder(
        multi_finder_cfg={
            "discover_models": False,
            "supported_models": {"sample": sample_module_id},
        }
    ) as finder:
        config = finder.find_model("sample")
        # Check RemoteModuleConfig has the right type, name, and task methods
        assert isinstance(config, RemoteModuleConfig)
        assert sample_module_id in config.module_id
        assert config.model_path == "sample"
        assert len(config.task_methods) == 1
        # Assert how many SampleTask methods there are
        assert len(config.task_methods[0][1]) == 4


def test_remote_finder_connection_template(sample_module_id):
    """Test to ensure that the connection can be a template"""
    hn_template = "foo.{}.svc"
    with temp_finder(
        connection_cfg={
            "hostname": hn_template,
            "port": 12345,
        },
        multi_finder_cfg={
            "discover_models": False,
            "supported_models": {
                "sample1": sample_module_id,
                "sample2": sample_module_id,
            },
        },
    ) as finder:
        for model_id in ["sample1", "sample2"]:
            config = finder.find_model(model_id)
            assert config.connection.hostname == hn_template.format(model_id)


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_multi_task_model(multi_task_model_id, open_port, protocol):
    """Test to ensure model finder works for models with multiple tasks"""
    with runtime_test_server(open_port, protocol=protocol) as server, temp_finder(
        connection_cfg={
            "hostname": "localhost",
            "port": server.port,
        },
        protocol=protocol,
    ) as finder:
        config = finder.find_model(multi_task_model_id)
        # Check RemoteModuleConfig has the right type, name, and task methods
        assert isinstance(config, RemoteModuleConfig)
        assert len(config.task_methods) == 3


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_discover_single_conn_models(
    sample_task_model_id, open_port, protocol
):
    """Test to ensure discovering models works for http"""
    with runtime_test_server(open_port, protocol=protocol) as server, temp_finder(
        connection_cfg={
            "hostname": "localhost",
            "port": server.port,
        },
        protocol=protocol,
    ) as finder:
        config = finder.find_model(sample_task_model_id)
        assert isinstance(config, RemoteModuleConfig)
        assert sample_task_model_id == config.model_path
        assert len(config.task_methods) == 1
        # Assert how many SampleTask methods there are
        assert len(config.task_methods[0][1]) == 4


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_discover_multi_conn_models(protocol):
    """Test to ensure discovery works with multiple servers"""
    hn_a = "foo.bar.com"
    hn_b = "baz.biz.com"
    port1 = 12345
    port2 = 23456
    mod_id_x = SampleModule.MODULE_ID
    mod_id_y = BoundingBoxModule.MODULE_ID
    model_id1 = random_test_id()
    model_id2 = random_test_id()
    model_id3 = random_test_id()
    model_id4 = random_test_id()

    class MockChannelSession:
        def __init__(self, *_, target: Optional[str] = None, **__):
            self.target = target

        @staticmethod
        def _get_resp(target: str):
            return {
                # hn A / port1 -> model1, model2
                f"{hn_a}:{port1}": ModelInfoResponse(
                    [
                        ModelInfo(name=model_id1, module_id=mod_id_x),
                        ModelInfo(name=model_id2, module_id=mod_id_x),
                    ]
                ),
                # hn A / port2 -> model3
                f"{hn_a}:{port2}": ModelInfoResponse(
                    [
                        ModelInfo(name=model_id3, module_id=mod_id_y),
                    ]
                ),
                # hn B / port3 -> model4
                f"{hn_b}:{port2}": ModelInfoResponse(
                    [
                        ModelInfo(name=model_id4, module_id=mod_id_y),
                    ]
                ),
            }.get(target)

        def get(self, target: str):
            resp_mock = MagicMock()
            resp = self._get_resp(target.split("/")[2])
            if not resp:
                resp_mock.status_code = 404
            else:
                resp_mock.status_code = 200
                resp_mock.json = MagicMock(return_value=resp.to_dict())
            return resp_mock

        def unary_unary(self, *_, **__):
            assert self.target
            resp = self._get_resp(self.target)
            if not resp:
                return MagicMock(side_effect=grpc.RpcError)
            return MagicMock(return_value=resp.to_proto())

    @contextmanager
    def mock_construct_grpc_channel(target, *_, **__):
        yield MockChannelSession(target=target)

    with patch(
        "caikit.runtime.client.remote_model_finder.construct_grpc_channel",
        new=mock_construct_grpc_channel,
    ), patch(
        "caikit.runtime.client.remote_model_finder.construct_requests_session",
        new=MockChannelSession,
    ):
        with temp_finder(
            remote_connections_cfg=[
                {"hostname": hn_a, "port": port1},
                {"hostname": hn_a, "port": port2},
                {"hostname": hn_b, "port": port2},
            ],
            protocol=protocol,
        ) as finder:
            # hn A / port1 -> model1, model2
            config1 = finder.find_model(model_id1)
            assert config1
            assert config1.connection.hostname == hn_a
            assert config1.connection.port == port1
            config2 = finder.find_model(model_id2)
            assert config2
            assert config2.connection.hostname == hn_a
            assert config2.connection.port == port1
            # hn A / port2 -> model3
            config3 = finder.find_model(model_id3)
            assert config3
            assert config3.connection.hostname == hn_a
            assert config3.connection.port == port2
            # hn B / port3 -> model4
            config4 = finder.find_model(model_id4)
            assert config4
            assert config4.connection.hostname == hn_b
            assert config4.connection.port == port2
            # Unknown model
            assert finder.find_model("unknown") is None


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_discover_mtls_models(sample_task_model_id, open_port, protocol):
    """Test to ensure discovering models works for https with MTLS and secure CA"""
    with generate_tls_configs(
        open_port, tls=True, mtls=True
    ) as config_overrides:  # noqa: SIM117
        with runtime_test_server(
            open_port,
            protocol=protocol,
            tls_config_override=config_overrides if protocol == "http" else None,
        ) as server_with_tls, temp_finder(
            connection_cfg={
                "hostname": "localhost",
                "port": server_with_tls.port,
                "tls": {
                    "enabled": True,
                    "ca_file": config_overrides["use_in_test"]["ca_cert"],
                    "cert_file": config_overrides["use_in_test"]["client_cert"],
                    "key_file": config_overrides["use_in_test"]["client_key"],
                },
            },
            protocol=protocol,
        ) as finder:
            config = finder.find_model(sample_task_model_id)
            assert isinstance(config, RemoteModuleConfig)
            assert sample_task_model_id == config.model_path
            assert len(config.task_methods) == 1
            # Assert how many SampleTask methods there are
            assert len(config.task_methods[0][1]) == 4


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_fail_ca_check(sample_task_model_id, open_port, protocol):
    """Test to ensure discovering models fails when the client doesn't trust the CA"""
    with generate_tls_configs(
        open_port, tls=True, mtls=False
    ) as config_overrides:  # noqa: SIM117
        with runtime_test_server(
            open_port,
            protocol=protocol,
            tls_config_override=config_overrides if protocol == "http" else None,
        ) as server_with_tls, temp_finder(
            connection_cfg={
                "hostname": "localhost",
                "port": server_with_tls.port,
                "tls": {
                    "enabled": True,
                    "insecure_verify": False,
                },
            },
            protocol=protocol,
        ) as finder:
            assert not finder.find_model(sample_task_model_id)


def test_remote_finder_discover_https_insecure_models(sample_task_model_id, open_port):
    """Test to ensure discovering models works for https without checking certs"""
    with generate_tls_configs(
        open_port, tls=True, mtls=False
    ) as config_overrides:  # noqa: SIM117
        with runtime_test_server(
            open_port,
            protocol="http",
            tls_config_override=config_overrides,
        ) as server_with_tls, temp_finder(
            connection_cfg={
                "hostname": "localhost",
                "port": server_with_tls.port,
                "tls": {"enabled": True, "insecure_verify": True},
            },
            protocol="http",
        ) as finder:
            config = finder.find_model(sample_task_model_id)
            assert isinstance(config, RemoteModuleConfig)
            assert sample_task_model_id == config.model_path
            assert len(config.task_methods) == 1
            # Assert how many SampleTask methods there are
            assert len(config.task_methods[0][1]) == 4


def test_remote_finder_discover_grpc_insecure_models():
    """Test to ensure discovering models raises an error when using insecure grpc"""
    with pytest.raises(ValueError):
        RemoteModelFinder(
            Config(
                {
                    "connection": {
                        "hostname": "localhost",
                        "port": 80,
                        "tls": {"enabled": True, "insecure_verify": True},
                    },
                    "protocol": "grpc",
                }
            ),
            "remote_finder",
        )


def test_remote_finder_not_found():
    """Test to ensure error is raised when no model is found"""
    with temp_finder(  # noqa: SIM117
        multi_finder_cfg={"discover_models": False, "supported_models": {"wrong": "id"}}
    ) as finder:
        assert not finder.find_model("sample")


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_lazy_discover_models(
    sample_task_model_id, open_port, protocol, box_model_path
):
    """Test to ensure lazily discovering models"""
    file_model_id = random_test_id()
    with runtime_test_server(open_port, protocol=protocol) as server, temp_finder(
        connection_cfg={
            "hostname": "localhost",
            "port": server.port,
        },
        protocol=protocol,
    ) as finder:
        config: RemoteModuleConfig | None = finder.find_model(sample_task_model_id)
        assert config
        assert isinstance(config, RemoteModuleConfig)
        assert sample_task_model_id == config.model_path

        # Assert file model hasn't been found
        assert not finder.find_model(file_model_id)

        with file_task_model_context(box_model_path, file_model_id):
            # Assert finder can find model once in context
            config = finder.find_model(model_path=file_model_id)
            assert config
            assert isinstance(config, RemoteModuleConfig)
            assert config.model_path == file_model_id


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_finder_lazy_discover_models_poll_time(
    sample_task_model_id, open_port, protocol, box_model_path
):
    """Test to ensure lazily discovering models doesn't work with poll time"""
    file_model_id = random_test_id()
    with runtime_test_server(open_port, protocol=protocol) as server, temp_finder(
        connection_cfg={
            "hostname": "localhost",
            "port": server.port,
        },
        min_poll_time=10,
        protocol=protocol,
    ) as finder:
        config: RemoteModuleConfig | None = finder.find_model(sample_task_model_id)
        assert config
        assert isinstance(config, RemoteModuleConfig)
        assert sample_task_model_id == config.model_path

        # Assert file model hasn't been found
        assert not finder.find_model(file_model_id)

        with file_task_model_context(box_model_path, file_model_id):
            # Assert finder still can't find model since it was checked to recently
            assert not finder.find_model(model_path=file_model_id)
