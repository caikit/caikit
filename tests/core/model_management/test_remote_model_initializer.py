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
Tests for the RemoteModelInitializer
"""

# Third Party
import pytest

# First Party
from aconfig import Config

# Local
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.core.model_management.remote_model_initializer import RemoteModelInitializer
from caikit.core.modules import ModuleBase, RemoteModuleConfig
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import get_train_params, get_train_request
from sample_lib.data_model import SampleInputType, SampleOutputType, SampleTrainingType
from sample_lib.modules.sample_task.sample_implementation import SampleModule
from tests.conftest import random_test_id
from tests.fixtures import Fixtures  # noqa: F401
from tests.runtime.conftest import multi_task_model_id  # noqa: F401
from tests.runtime.conftest import open_port  # noqa: F401
from tests.runtime.conftest import sample_task_model_id  # noqa: F401
from tests.runtime.conftest import generate_tls_configs, runtime_test_server
import caikit

## Tests #######################################################################


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_initializer_insecure_predict(sample_task_model_id, open_port, protocol):
    """Test to ensure RemoteModule Initializer works for insecure connections"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )

    # Construct Remote Module Config
    connection_info = {"hostname": "localhost", "port": open_port, "protocol": protocol}
    remote_config = RemoteModuleConfig.load_from_module(
        local_module_class, connection_info, sample_task_model_id
    )
    # Set random module_id so tests don't conflict
    remote_config.module_id = random_test_id()

    with runtime_test_server(open_port, protocol=protocol):
        remote_initializer = RemoteModelInitializer(Config({}), "test")
        remote_model = remote_initializer.init(remote_config)
        assert isinstance(remote_model, ModuleBase)

        model_result = remote_model.run(SampleInputType(name="Test"), throw=False)
        assert isinstance(model_result, SampleOutputType)
        assert model_result.greeting == "Hello Test"


# Input streaming is only supported on grpc
@pytest.mark.parametrize("protocol", ["grpc"])
def test_remote_initializer_input_streaming(sample_task_model_id, open_port, protocol):
    """Test to ensure Remote Initializer works with input streaming"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with runtime_test_server(open_port, protocol=protocol):
        # Construct Remote Module Config
        connection_info = {
            "hostname": "localhost",
            "port": open_port,
            "protocol": protocol,
        }
        remote_config = RemoteModuleConfig.load_from_module(
            local_module_class, connection_info, sample_task_model_id
        )
        # Set random module_id so tests don't conflict
        remote_config.module_id = random_test_id()

        remote_model = remote_initializer.init(remote_config)
        assert isinstance(remote_model, ModuleBase)

        stream_input = DataStream.from_iterable(
            [
                SampleInputType(name="Test1"),
                SampleInputType(name="Test2"),
                SampleInputType(name="Test3"),
            ]
        )
        model_result = remote_model.run_stream_in(stream_input, greeting="Hello Tests ")
        assert isinstance(model_result, SampleOutputType)
        assert model_result.greeting == "Hello Tests Test1,Test2,Test3"


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_initializer_output_streaming(sample_task_model_id, open_port, protocol):
    """Test to ensure Remote Initializer works when streaming outputs"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with runtime_test_server(open_port, protocol=protocol):
        # Construct Remote Module Config
        connection_info = {
            "hostname": "localhost",
            "port": open_port,
            "protocol": protocol,
        }
        remote_config = RemoteModuleConfig.load_from_module(
            local_module_class, connection_info, sample_task_model_id
        )
        # Set random module_id so tests don't conflict
        remote_config.module_id = random_test_id()

        remote_model = remote_initializer.init(remote_config)
        assert isinstance(remote_model, ModuleBase)

        model_result = remote_model.run_stream_out(
            SampleInputType(name="Test"), err_stream=False
        )
        assert isinstance(model_result, DataStream)
        stream_results = [item for item in model_result]
        assert len(stream_results) == 10
        for item in stream_results:
            assert item.greeting == "Hello Test stream"


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_initializer_streaming_deleted_model(
    sample_task_model_id, open_port, protocol
):
    """Test to ensure Remote Initializer is still able to stream outputs after the RemoteModelBase
    has been deleted/out of scope"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with runtime_test_server(open_port, protocol=protocol):
        # Construct Remote Module Config
        connection_info = {
            "hostname": "localhost",
            "port": open_port,
            "protocol": protocol,
        }
        remote_config = RemoteModuleConfig.load_from_module(
            local_module_class, connection_info, sample_task_model_id
        )
        # Set random module_id so tests don't conflict
        remote_config.module_id = random_test_id()

        remote_model = remote_initializer.init(remote_config)
        assert isinstance(remote_model, ModuleBase)

        model_result = remote_model.run_stream_out(
            SampleInputType(name="Test"), err_stream=False
        )
        assert isinstance(model_result, DataStream)

        # Delete Model Object
        del remote_model

        # Assert stream can still be read
        stream_results = [item for item in model_result]
        assert len(stream_results) == 10
        for item in stream_results:
            assert item.greeting == "Hello Test stream"


# Only GRPC Supports bidi streams
@pytest.mark.parametrize("protocol", ["grpc"])
def test_remote_initializer_input_output_streaming(
    sample_task_model_id, open_port, protocol
):
    """Test to ensure Remote Initializer works when streaming outputs"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with runtime_test_server(open_port, protocol=protocol):
        # Construct Remote Module Config
        connection_info = {
            "hostname": "localhost",
            "port": open_port,
            "protocol": protocol,
        }
        remote_config = RemoteModuleConfig.load_from_module(
            local_module_class, connection_info, sample_task_model_id
        )
        # Set random module_id so tests don't conflict
        remote_config.module_id = random_test_id()

        remote_model = remote_initializer.init(remote_config)
        assert isinstance(remote_model, ModuleBase)

        stream_input = DataStream.from_iterable(
            [
                SampleInputType(name="Test1"),
                SampleInputType(name="Test2"),
                SampleInputType(name="Test3"),
            ]
        )
        model_result = remote_model.run_bidi_stream(stream_input)
        assert isinstance(model_result, DataStream)
        stream_results = [item.greeting for item in model_result]
        assert len(stream_results) == 3
        assert stream_results == [
            "Hello Test1",
            "Hello Test2",
            "Hello Test3",
        ]


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_initializer_train(sample_task_model_id, open_port, protocol):
    """Test to ensure Remote Initializer works when streaming outputs"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with runtime_test_server(open_port, protocol=protocol):
        # Construct Remote Module Config
        connection_info = {
            "hostname": "localhost",
            "port": open_port,
            "protocol": protocol,
        }
        remote_config = RemoteModuleConfig.load_from_module(
            local_module_class, connection_info, sample_task_model_id
        )
        # Set random module_id so tests don't conflict
        remote_config.module_id = random_test_id()

        remote_model = remote_initializer.init(remote_config)
        assert isinstance(remote_model, ModuleBase)

        stream_type = (
            caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
        )
        training_data = stream_type(
            data_stream=stream_type.JsonData(
                data=[SampleTrainingType(1), SampleTrainingType(2)]
            )
        )

        model_result = remote_model.train(
            training_data=training_data, union_list=["str", "sequence"]
        )
        assert isinstance(model_result, ModuleBase)


@pytest.mark.parametrize("protocol", ["grpc", "http"])
def test_remote_initializer_mtls_predict(sample_task_model_id, open_port, protocol):
    """Test to ensure Remote Initializer works with TLS and MTLS"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with generate_tls_configs(open_port, tls=True, mtls=True) as config_overrides:
        with runtime_test_server(
            open_port,
            protocol=protocol,
            tls_config_override=config_overrides if protocol == "http" else None,
        ):
            # Construct Remote Module Config
            connection_info = {
                "hostname": "localhost",
                "port": open_port,
                "protocol": protocol,
                "tls": {
                    "enabled": True,
                    "ca_file": config_overrides["use_in_test"]["ca_cert"],
                    "cert_file": config_overrides["use_in_test"]["client_cert"],
                    "key_file": config_overrides["use_in_test"]["client_key"],
                },
            }
            remote_config = RemoteModuleConfig.load_from_module(
                local_module_class, connection_info, sample_task_model_id
            )
            # Set random module_id so tests don't conflict
            remote_config.module_id = random_test_id()

            remote_model = remote_initializer.init(remote_config)
            assert isinstance(remote_model, ModuleBase)

            model_result = remote_model.run(SampleInputType(name="Test"))
            assert isinstance(model_result, SampleOutputType)
            assert "Hello Test" == model_result.greeting


def test_remote_initializer_https_unverified_predict(sample_task_model_id, open_port):
    """Test to ensure RemoteModuleInitializer works with an unverified connection over HTTPS"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with generate_tls_configs(open_port, tls=True, mtls=False) as config_overrides:
        with runtime_test_server(
            open_port,
            protocol="http",
            tls_config_override=config_overrides,
        ):
            # Construct Remote Module Config
            connection_info = {
                "hostname": "localhost",
                "port": open_port,
                "protocol": "http",
                "tls": {
                    "enabled": True,
                    "insecure_verify": True,
                },
            }
            remote_config = RemoteModuleConfig.load_from_module(
                local_module_class, connection_info, sample_task_model_id
            )
            # Set random module_id so tests don't conflict
            remote_config.module_id = random_test_id()

            remote_model = remote_initializer.init(remote_config)
            assert isinstance(remote_model, ModuleBase)

            model_result = remote_model.run(SampleInputType(name="Test"))
            assert isinstance(model_result, SampleOutputType)
            assert model_result.greeting == "Hello Test"


def test_remote_initializer_grpc_unverified_predict(sample_task_model_id, open_port):
    """Test to ensure RemoteModuleInitializer raises an error when unverified GRPC is enabled"""
    local_module_class = (
        ModelManager.get_instance().retrieve_model(sample_task_model_id).__class__
    )
    remote_initializer = RemoteModelInitializer(Config({}), "test")

    with generate_tls_configs(open_port, tls=True, mtls=False):
        with runtime_test_server(open_port, protocol="grpc"):
            # Construct Remote Module Config
            connection_info = {
                "hostname": "localhost",
                "port": open_port,
                "protocol": "grpc",
                "tls": {
                    "enabled": True,
                    "insecure_verify": True,
                },
            }
            remote_config = RemoteModuleConfig.load_from_module(
                local_module_class, connection_info, sample_task_model_id
            )
            # Set random module_id so tests don't conflict
            remote_config.module_id = random_test_id()

            with pytest.raises(ValueError):
                remote_initializer.init(remote_config)
