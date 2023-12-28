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
                connection:
                    hostname: str <remote host>
                    port: Optional[int]=80/443 <remote port>
                    protocol: Optional[str]="grpc" <protocol the remote server is using (grpc or http)>
                    tls:
                        enabled: Optional[bool]=False <if ssl is enabled on the remote server>
                        ca_file: Optional[str]=None <path to remote ca file>
                        cert_file: Optional[str]=None <path to MTLS cert>
                        key_file: Optional[str]=None <path to MTLS key>
                        insecure_verify: Optional[bool]=False <If client should validate server remote CA>
                    options:  Optional[Dict[str,str]]={} <optional dict of grpc or http configuration options>
                    timeout: Optional[int]=None <Optional timeout setting for remote connections>
                    model_key: Optional[str]=MODEL_MESH_MODEL_ID_KEY <Optional setting to override the grpc model name>
                discover_models: Optional[bool]=True <bool to automatically discover remote models via the /info/models endpoint>
                supported_models: Optional[Dict[str, str]]={} <mapping of model names to module_ids that this remote supports>
                    <model_path>: <module_id>

"""
# Standard
from http.client import HTTP_PORT, HTTPS_PORT
from pathlib import Path
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
from caikit.interfaces.runtime.data_model import ModelInfoRequest, ModelInfoResponse
from caikit.interfaces.runtime.server import (
    MODELS_INFO_ENDPOINT,
    ServiceType,
    get_grpc_route_name,
)
from caikit.interfaces.runtime.service import MODEL_MESH_MODEL_ID_KEY

log = alog.use_channel("RFIND")
error = error_handler.get(log)


### Finder Definitions


class RemoteModelFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with an optional path prefix"""

        self._instance_name = instance_name
        self._discover_models = config.get("discover_models", True)
        self._supported_models = config.get("supported_models", {})

        # Type/Value check connection parameters
        self._connection = config.connection
        self._hostname = self._connection.get("hostname")
        self._port = self._connection.get("port")
        self._protocol = self._connection.get("protocol", "grpc")
        self._timeout = self._connection.get("timeout")
        self._model_key = self._connection.get("model_key", MODEL_MESH_MODEL_ID_KEY)
        self._options = self._connection.get("options", {})
        error.value_check(
            "<COR72281545E>",
            self._protocol in ["grpc", "http"],
            "Unknown protocol: %s",
            self._protocol,
        )
        error.type_check(
            "<COR74343245E>",
            dict,
            supported_models=self._supported_models,
            options=self._options,
        )
        error.type_check(
            "<COR72281587E>", str, host=self._hostname, model_key=self._model_key
        )
        error.type_check(
            "<COR73381567E>",
            int,
            allow_none=True,
            port=self._port,
            timeout=self._timeout,
        )

        # Type/Value check TLS parameters
        self._tls_info = self._connection.get("tls", {})
        self._tls_enabled = self._tls_info.get("enabled", False)
        self._tls_ca_file = self._tls_info.get("ca_file")
        self._tls_cert_file = self._tls_info.get("cert_file")
        self._tls_key_file = self._tls_info.get("key_file")
        self._tls_insecure_verify = self._tls_info.get("insecure_verify", False)
        error.type_check(
            "<COR74321567E>",
            str,
            allow_none=True,
            tls_ca=self._tls_ca_file,
            tls_cert=self._tls_cert_file,
            key_file=self._tls_key_file,
        )
        error.type_check(
            "COR74322567E",
            bool,
            tls_enabled=self._tls_enabled,
            insecure_verify=self._tls_insecure_verify,
        )

        if self._protocol == "grpc" and self._tls_enabled:
            error.value_check(
                "<COR74451567E>",
                not self._tls_insecure_verify,
                "GRPC does not support insecure TLS connections. Please provide a valid CA certificate",
            )

        # Set default port
        if not self._port:
            if self._tls_enabled:
                self._port = HTTPS_PORT
            else:
                self._port = HTTP_PORT

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

        target = f"{self._hostname}:{self._port}"
        options = tuple(self._options.items())

        # Generate GRPC Channel
        if self._tls_enabled:
            # Gather CA and MTLS data
            ca_data = None
            if self._tls_ca_file and Path(self._tls_ca_file).exists():
                ca_data = Path(self._tls_ca_file).read_bytes()

            mtls_key_data = None
            if self._tls_key_file and Path(self._tls_key_file).exists():
                mtls_key_data = Path(self._tls_key_file).read_bytes()

            mtls_cert_data = None
            if self._tls_cert_file and Path(self._tls_cert_file).exists():
                mtls_cert_data = Path(self._tls_cert_file).read_bytes()

            grpc_credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_data,
                private_key=mtls_key_data,
                certificate_chain=mtls_cert_data,
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
                    ModelInfoRequest().to_proto(), timeout=self._timeout
                )
            except grpc.RpcError as exc:
                log.warning(
                    "Unable to discover modules from remote: %s. Error: %s",
                    self._hostname,
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
        target = f"{self._hostname}:{self._port}{MODELS_INFO_ENDPOINT}"
        request_kwargs = {}
        if self._tls_enabled:
            target = f"https://{target}"

            # Configure the TLS CA settings
            if self._tls_insecure_verify:
                request_kwargs["verify"] = False
            else:
                if self._tls_ca_file:
                    request_kwargs["verify"] = self._tls_ca_file
                else:
                    request_kwargs["verify"] = True

            if self._tls_cert_file and self._tls_key_file:
                request_kwargs["cert"] = (self._tls_cert_file, self._tls_key_file)

        else:
            target = f"http://{target}"

        # Send HTTP Request
        resp = requests.get(target, **request_kwargs)
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
