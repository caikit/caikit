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
The RemoteModelFinder locates models that loaded by a remote runtime.

Configuration for RemoteModelFinder lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: REMOTE
            config:
                connection: ConnectionInfo
                model_key: Optional[str]=MODEL_MESH_MODEL_ID_KEY <Optional setting to override the grpc model name>
                protocol: Optional[str]="grpc" <protocol the remote server is using (grpc or http)>
                discover_models: Optional[bool]=True <bool to automatically discover remote models via the /info/models endpoint>
                supported_models: Optional[Dict[str, str]]={} <mapping of model names to module_ids that this remote supports>
                    <model_path>: <module_id>

"""
# Standard
from typing import Dict, Optional

# Third Party
import grpc
import requests

# First Party
import aconfig
import alog

# Local
from caikit.core.exceptions import error_handler
from caikit.core.model_management.model_finder_base import ModelFinderBase
from caikit.core.modules import RemoteModuleConfig
from caikit.interfaces.common.data_model.remote import ConnectionInfo, ConnectionTlsInfo
from caikit.interfaces.runtime.data_model import ModelInfoRequest, ModelInfoResponse
from caikit.runtime.names import (
    MODEL_MESH_MODEL_ID_KEY,
    MODELS_INFO_ENDPOINT,
    ServiceType,
    get_grpc_route_name,
)

log = alog.use_channel("RFIND")
error = error_handler.get(log)


### Finder Definitions


class RemoteModelFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with an optional path prefix"""

        self._instance_name = instance_name

        # Type/Value check connection parameters
        self._tls = config.connection["tls"] = ConnectionTlsInfo(
            **config.connection.get("tls", {})
        )
        self._connection = ConnectionInfo(**config.connection)

        # Type/Value check default parameters
        self._model_key = config.get("model_key", MODEL_MESH_MODEL_ID_KEY)
        error.type_check("<COR72281587E>", str, model_key=self._model_key)

        self._protocol = config.get("protocol", "grpc")
        error.value_check(
            "<COR72281545E>",
            self._protocol in ["grpc", "http"],
            "Unknown protocol: %s",
            self._protocol,
        )

        if self._protocol == "grpc" and self._tls.enabled:
            error.value_check(
                "<COR74451567E>",
                not self._tls.insecure_verify,
                "GRPC does not support insecure TLS connections. Please provide a valid CA certificate",
            )

        # Type/Value check model parameters
        self._discover_models = config.get("discover_models", True)
        self._supported_models = config.get("supported_models", {})
        error.type_check(
            "<COR74343245E>",
            dict,
            supported_models=self._supported_models,
        )
        error.type_check(
            "<COR74342245E>",
            bool,
            discover_models=self._discover_models,
        )

        # Discover remote models
        if self._discover_models:
            self._supported_models.update(self.discover_models())

    def find_model(
        self,
        model_path: str,
        **__,
    ) -> Optional[RemoteModuleConfig]:
        """Check if the remote runtime supports the model_path"""

        # If model_path is not one of the supported models then raise an error
        if model_path not in self._supported_models:
            raise KeyError(
                f"Model {model_path} is not supported by finder {self._instance_name}"
            )

        return RemoteModuleConfig.load_from_module(
            module_reference=self._supported_models.get(model_path),
            connection_info=self._connection,
            protocol=self._protocol,
            model_key=self._model_key,
            model_path=model_path,
        )

    def discover_models(self) -> Dict[str, str]:
        """Helper method to discover models from a remote
        runtime. This is a separate function to help with subclassing

        Returns:
            model_map: Dict[str, str]
                The map of models to modules
        """
        if self._protocol == "grpc":
            return self._discover_grpc_models()
        elif self._protocol == "http":
            return self._discover_http_models()

    ### Discovery Helper Functions

    def _discover_grpc_models(self) -> Dict[str, str]:
        """Helper function to get all the supported models and modules
        from a remote GRPC runtime

        Returns:
           support_models: Dict[str, str
               Mapping of remote model names to module ids
        """

        target = f"{self._connection.hostname}:{self._connection.port}"
        options = tuple(self._connection._options.items())

        # Generate GRPC Channel
        if self._tls.enabled:
            # Assert that TLS files exists
            if self._tls.ca_file and not self._tls.ca_file_data:
                raise FileNotFoundError(
                    f"Unable to find TLS CA File {self._tls.ca_file}"
                )
            if self._tls.key_file and not self._tls.key_file_data:
                raise FileNotFoundError(
                    f"Unable to find TLS Key File {self._tls.key_file}"
                )
            if self._tls.cert_file and not self._tls.cert_file_data:
                raise FileNotFoundError(
                    f"Unable to find TLS Cert File {self._tls.cert_file}"
                )

            # Gather CA and MTLS data
            grpc_credentials = grpc.ssl_channel_credentials(
                root_certificates=self._tls.ca_file_data,
                private_key=self._tls.key_file_data,
                certificate_chain=self._tls.cert_file_data,
            )

            # Construct secure channel
            channel_func = grpc.secure_channel
            channel_args = [target, grpc_credentials, options]
        else:
            channel_func = grpc.insecure_channel
            channel_args = [target, options]

        # Construct Info RPC and submit request
        log.debug2(
            "Constructing grpc finder channel with %s and %s",
            channel_func,
            str(channel_args),
        )
        with channel_func(*channel_args) as channel:
            info_service_rpc = channel.unary_unary(
                get_grpc_route_name(ServiceType.INFO, "GetModelsInfo"),
                request_serializer=ModelInfoRequest.get_proto_class().SerializeToString,
                response_deserializer=ModelInfoResponse.get_proto_class().FromString,
            )
            try:
                model_info_proto = info_service_rpc(
                    ModelInfoRequest().to_proto(), timeout=self._connection.timeout
                )
            except grpc.RpcError as exc:
                log.warning(
                    "Unable to discover modules from remote: %s. Error: %s",
                    self._connection.hostname,
                    str(exc),
                )
                return {}

        model_info_response = ModelInfoResponse.from_proto(model_info_proto)

        # Parse response into dictionary of name->id
        supported_modules = {}
        for model_info in model_info_response.models:
            model_name = model_info.name
            module_id = model_info.module_id

            log.debug(
                "Discovered model %s with module_id %s from remote runtime",
                model_name,
                module_id,
            )
            supported_modules[model_name] = module_id

        return supported_modules

    def _discover_http_models(self) -> Dict[str, str]:
        """Helper function to get all the supported models and modules
        from a remote HTTP runtime

        Returns:
            supported_models:Dict[str, str]
                Mapping of remote model names to module_ids
        """

        # Configure HTTP Client object
        target = (
            f"{self._connection.hostname}:{self._connection.port}{MODELS_INFO_ENDPOINT}"
        )
        request_kwargs = {}
        if self._tls.enabled:
            target = f"https://{target}"

            # Configure the TLS CA settings
            if self._tls.insecure_verify:
                request_kwargs["verify"] = False
            else:
                if self._tls.ca_file:
                    request_kwargs["verify"] = self._tls.ca_file
                else:
                    request_kwargs["verify"] = True

            if self._tls.cert_file and self._tls.key_file:
                request_kwargs["cert"] = (self._tls.cert_file, self._tls.key_file)

        else:
            target = f"http://{target}"

        # Send HTTP Request
        try:
            resp = requests.get(target, **request_kwargs)
        except requests.exceptions.RequestException as exc:
            log.warning(
                "Unable to discover modules from remote: %s. Error: %s",
                self._connection.hostname,
                str(exc),
            )
            return {}

        if resp.status_code != 200:
            log.warning(
                "Unable to discover modules from remote: %s. Error: %s",
                target,
                resp.reason,
            )
            return {}

        # Load the response as a json object
        model_info = resp.json()

        # Parse response into dictionary of name->id
        supported_modules = {}
        for model_dict in model_info.get("models", []):
            model_name = model_dict.get("name")
            module_id = model_dict.get("module_id")

            log.debug(
                "Discovered model %s with module_id %s from remote runtime",
                model_name,
                module_id,
            )
            supported_modules[model_name] = module_id

        return supported_modules
