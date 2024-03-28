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
The RemoteModelFinder locates models that are loaded in a remote runtime.

Configuration for RemoteModelFinder lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: REMOTE
            config:
                connection: ConnectionInfo <Required Connection Information>
                model_key: Optional[str]=MODEL_MESH_MODEL_ID_KEY <Optional setting to override the
                    grpc model name>
                protocol: Optional[str]="grpc" <protocol the remote server is using (grpc or http)>
                min_poll_time: Optional[int]=30 <minimum time before attempting to rediscover
                    models>
                discover_models: Optional[bool]=True <bool to automatically discover remote models
                    via the /info/models endpoint>
                supported_models: Optional[Dict[str, str]]={} <mapping of model names to module_ids
                    that this remote supports. This is automatically populated by discover_models>
                    <model_path>: <module_id>

"""
# Standard
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional

# Third Party
from requests import RequestException
import grpc

# First Party
import aconfig
import alog

# Local
from caikit.core.exceptions import error_handler
from caikit.core.model_management.factories import model_finder_factory
from caikit.core.model_management.model_finder_base import ModelFinderBase
from caikit.interfaces.common.data_model.remote import ConnectionInfo
from caikit.interfaces.runtime.data_model import ModelInfoRequest, ModelInfoResponse
from caikit.runtime.client.remote_config import RemoteModuleConfig
from caikit.runtime.client.utils import (
    construct_grpc_channel,
    construct_requests_session,
)
from caikit.runtime.names import (
    MODEL_MESH_MODEL_ID_KEY,
    MODELS_INFO_ENDPOINT,
    ServiceType,
    get_grpc_route_name,
)

log = alog.use_channel("RFIND")
error = error_handler.get(log)


### Finder Definitions


@dataclass
class ModuleConnectionInfo:
    conn: ConnectionInfo
    module_id: str


class RemoteModelFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "REMOTE"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with a config and instance name"""

        self._instance_name = instance_name

        # Initialize model_name -> connection map
        self._connections: Dict[str, ConnectionInfo] = {}
        self._connection_template: Optional[ConnectionInfo] = None

        # If a remote_models key is found, it's a mapping from model name to
        # connection info
        for remote_conn in config.get("remote_connections", []):
            conn = ConnectionInfo(**remote_conn)
            self._connections[f"{conn.hostname}:{conn.port}"] = conn

        # If a single "global" connection given, initialize with model_name None
        if config.connection:
            default_conn = ConnectionInfo(**config.connection)
            if f"{default_conn.hostname}:{default_conn.port}" not in self._connections:
                self._connection_template = default_conn
                self._connections[default_conn.hostname] = default_conn

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

        if self._protocol == "grpc":
            for conn in self._connections.values():
                error.value_check(
                    "<COR74451567E>",
                    not conn.tls.enabled or not conn.tls.insecure_verify,
                    "GRPC does not support insecure TLS connections."
                    "Please provide a valid CA certificate",
                )

        # Initialize the supported models using the model connection info
        self._supported_models: Dict[str, ModuleConnectionInfo] = {}
        supported_models = config.get("supported_models") or {}
        error.value_check(
            "<NLP77334255E>",
            not supported_models or self._connection_template,
            "Cannot provide 'supported_models' without 'connection'",
        )
        for model_name, module_id in supported_models.items():
            if model_conn := self._render_conn_template(model_name):
                self._supported_models[model_name] = ModuleConnectionInfo(
                    model_conn, module_id
                )

        # Type/Value check model parameters
        self._discover_models = config.get("discover_models", True)
        self._min_poll_time = config.get("min_poll_time", 30)
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
        error.type_check("", int, min_poll_time=self._min_poll_time)

        # If discovery models is enabled construct lock objects
        # and then run discovery
        if self._discover_models:
            self._last_discovered_time = None
            self._poll_delta = timedelta(seconds=self._min_poll_time)
            self._discovery_lock = Lock()
            self._supported_models.update(self._discover())

    def find_model(
        self,
        model_path: str,
        **__,
    ) -> Optional[RemoteModuleConfig]:
        """Check if the remote runtime supports the model_path"""

        # If model_path is not detected and discover models is enabled attempt
        # rediscovery
        if model_path not in self._supported_models and self._discover_models:
            self._safe_discover(model_path)

        # If model_path is not one of the supported models then raise an error
        if model_path not in self._supported_models:
            log.debug(
                "Model %s is not supported by finder %s",
                model_path,
                self._instance_name,
            )
            return

        module_conn_info = self._supported_models.get(model_path)
        return RemoteModuleConfig.load_from_module(
            module_reference=module_conn_info.module_id,
            connection_info=module_conn_info.conn,
            protocol=self._protocol,
            model_key=self._model_key,
            model_path=model_path,
        )

    ### Discovery Helper Functions

    def _discover(
        self, model_name: Optional[str] = None
    ) -> Dict[str, ModuleConnectionInfo]:
        """Helper method to discover models from a remote
        runtime. This is a separate function to help with subclassing

        Returns:
            model_map: Dict[str, str]
                The map of models to modules
        """
        error.value_check(
            "<COR45835854E>",
            self._protocol in ["grpc", "http"],
            "Invalid protocol: {}",
            self._protocol,
        )
        if self._protocol == "grpc":
            return self._discover_grpc_models(model_name)
        return self._discover_http_models(model_name)

    def _safe_discover(
        self, model_name: Optional[str] = None
    ) -> Dict[str, ModuleConnectionInfo]:
        """Helper function that lazily discovers models in a
        thread safe manor. This function also ensures we don't overload
        the remote server with discovery requests

        Returns:
            Dict[str, str]: Result of discover_models
        """
        with self._discovery_lock:
            current_time = datetime.now()

            # If discovery was ran recently then return the cached results
            if (
                self._last_discovered_time
                and self._last_discovered_time + self._poll_delta > current_time
            ):
                return self._supported_models

            # Run discovery
            self._last_discovered_time = current_time
            self._supported_models = self._discover(model_name)
            return self._supported_models

    def _discover_grpc_models(
        self,
        model_name: Optional[str],
    ) -> Dict[str, ModuleConnectionInfo]:
        """Helper function to get all the supported models and modules
        from a remote GRPC runtime

        Returns:
           support_models: Dict[str, str
               Mapping of remote model names to module ids
        """
        supported_modules = {}
        for conn in self._get_conn_candidates(model_name):
            target = f"{conn.hostname}:{conn.port}"
            options = [tuple(opt) for opt in conn.options.items()]
            with construct_grpc_channel(target, options, conn.tls) as channel:
                info_service_rpc = channel.unary_unary(
                    get_grpc_route_name(ServiceType.INFO, "GetModelsInfo"),
                    request_serializer=ModelInfoRequest.get_proto_class().SerializeToString,
                    response_deserializer=ModelInfoResponse.get_proto_class().FromString,
                )
                try:
                    model_info_proto = info_service_rpc(
                        ModelInfoRequest().to_proto(), timeout=conn.timeout
                    )

                    model_info_response = ModelInfoResponse.from_proto(model_info_proto)

                    # Parse response into dictionary of name->conn
                    for model_info in model_info_response.models:
                        model_name = model_info.name
                        module_id = model_info.module_id

                        log.debug(
                            "Discovered model %s with module_id %s from remote runtime %s",
                            model_name,
                            module_id,
                            target,
                        )
                        # NOTE: If multiple servers support the same model, the
                        #   first to be checked will win
                        supported_modules.setdefault(
                            model_name, ModuleConnectionInfo(conn, module_id)
                        )
                except grpc.RpcError as exc:
                    log.warning(
                        "Unable to discover modules from remote: %s. Error: %s",
                        target,
                        str(exc),
                    )

        return supported_modules

    def _discover_http_models(
        self,
        model_name: Optional[str],
    ) -> Dict[str, ConnectionInfo]:
        """Helper function to get all the supported models and modules
        from a remote HTTP runtime

        Returns:
            supported_models:Dict[str, str]
                Mapping of remote model names to module_ids
        """
        supported_modules = {}
        for conn in self._get_conn_candidates(model_name):

            # Configure HTTP target and Session object
            http_scheme = "https://" if conn.tls.enabled else "http://"
            target = (
                f"{http_scheme}{conn.hostname}:" f"{conn.port}{MODELS_INFO_ENDPOINT}"
            )
            session = construct_requests_session(conn.options, conn.tls, conn.timeout)

            # Send HTTP Request
            try:
                resp = session.get(target)

                if resp.status_code != 200:
                    log.warning(
                        "Unable to discover modules from remote: %s. Error: %s",
                        target,
                        resp.reason,
                    )
                else:

                    # Load the response as a json object
                    model_info = resp.json()

                    # Parse response into dictionary of name->id
                    for model_dict in model_info.get("models", []):
                        model_name = model_dict.get("name")
                        module_id = model_dict.get("module_id")

                        log.debug(
                            "Discovered model %s with module_id %s from remote runtime",
                            model_name,
                            module_id,
                        )
                        # NOTE: If multiple servers support the same model, the
                        #   first to be checked will win
                        supported_modules.setdefault(
                            model_name, ModuleConnectionInfo(conn, module_id)
                        )
            except RequestException as exc:
                log.warning(
                    "Unable to discover modules from remote: %s. Error: %s",
                    target,
                    str(exc),
                )

        return supported_modules

    def _render_conn_template(self, model_name: str) -> Optional[ConnectionInfo]:
        """Common utility to get the connection for a given model"""
        if self._connection_template is not None:
            conn_dict = self._connection_template.to_dict()
            conn_dict["hostname"] = self._connection_template.hostname.format(
                model_name
            )
            return ConnectionInfo(**conn_dict)

    def _get_conn_candidates(self, model_name: Optional[str]) -> List[ConnectionInfo]:
        """Common utility to get all connections to try"""
        candidate_conns = []
        if (
            model_name is not None
            and self._connection_template is not None
            and (model_conn := self._render_conn_template(model_name))
        ):
            candidate_conns.append(model_conn)
        candidate_conns.extend(self._connections.values())
        return candidate_conns


# Register the remote finder once it has been constructed
model_finder_factory.register(RemoteModelFinder)
