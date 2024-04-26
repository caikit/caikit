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
# Standard
from concurrent import futures
from typing import Optional, Union
import os

# Third Party
from grpc_health.v1 import health, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from py_grpc_prometheus.prometheus_server_interceptor import PromServerInterceptor
import grpc

# First Party
import aconfig
import alog

# Local
# Get the injectable servicer class definitions
from caikit import get_config
from caikit.runtime.interceptors.caikit_runtime_server_wrapper import (
    CaikitRuntimeServerWrapper,
)
from caikit.runtime.protobufs import (
    model_runtime_pb2,
    model_runtime_pb2_grpc,
    process_pb2_grpc,
)
from caikit.runtime.server_base import RuntimeServerBase
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.servicers.info_servicer import InfoServicer
from caikit.runtime.servicers.model_management_servicer import (
    ModelManagementServicerImpl,
)
from caikit.runtime.servicers.model_runtime_servicer import ModelRuntimeServicerImpl
from caikit.runtime.servicers.model_train_servicer import ModelTrainServicerImpl
from caikit.runtime.servicers.training_management_servicer import (
    TrainingManagementServicerImpl,
)

# Have pylint ignore broad exception catching in this file so that we can log all
# unexpected errors using alog.
# pylint: disable=W0703
import caikit.core.data_model

log = alog.use_channel("SERVR-GRPC")
PROMETHEUS_METRICS_INTERCEPTOR = PromServerInterceptor()


class RuntimeGRPCServer(RuntimeServerBase):
    """An implementation of a gRPC server that serves caikit runtimes"""

    def __init__(
        self,
        tls_config_override: Optional[aconfig.Config] = None,
    ):
        super().__init__(get_config().runtime.grpc.port, tls_config_override)

        # Initialize basic server
        self.server = grpc.server(
            thread_pool=self.thread_pool,
            interceptors=(PROMETHEUS_METRICS_INTERCEPTOR,),
            options=(self.config.runtime.grpc.options or {}).items(),
        )

        # Start metrics server
        RuntimeServerBase._start_metrics_server()

        # Start tracking service names for reflection
        service_names = [reflection.SERVICE_NAME]

        # Intercept an Inference Service
        self._global_predict_servicer = None
        self.model_management_service = None
        self.training_management_service = None
        if self.enable_inference:
            log.info("<RUN20247875I>", "Enabling gRPC inference service")
            self._global_predict_servicer = GlobalPredictServicer(
                self.inference_service, interrupter=self.interrupter
            )
            self.server = CaikitRuntimeServerWrapper(
                server=self.server,
                global_predict=self._global_predict_servicer.Predict,
                intercepted_svc_package=self.inference_service,
            )
            service_names.append(self.inference_service.descriptor.full_name)

            # Register inference service
            self.inference_service.registration_function(
                self.inference_service.service, self.server
            )

            # Register model management service
            self.model_management_service: ServicePackage = (
                ServicePackageFactory.get_service_package(
                    ServicePackageFactory.ServiceType.MODEL_MANAGEMENT,
                )
            )
            service_names.append(self.model_management_service.descriptor.full_name)
            self.model_management_service.registration_function(
                ModelManagementServicerImpl(), self.server
            )

        # And intercept a training service, if we have one
        if self.enable_training and self.training_service:
            log.info("<RUN20247827I>", "Enabling gRPC training service")
            global_train_servicer = GlobalTrainServicer(self.training_service)
            self.server = CaikitRuntimeServerWrapper(
                server=self.server,
                global_predict=global_train_servicer.Train,
                intercepted_svc_package=self.training_service,
            )
            service_names.append(self.training_service.descriptor.full_name)

            # Register training service
            self.training_service.registration_function(
                self.training_service.service, self.server
            )

            # Add model train servicer to the gRPC server
            process_pb2_grpc.add_ProcessServicer_to_server(
                ModelTrainServicerImpl(self.training_service), self.server
            )

            # Add training management servicer to the gRPC server
            self.training_management_service: ServicePackage = (
                ServicePackageFactory.get_service_package(
                    ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
                )
            )
            service_names.append(self.training_management_service.descriptor.full_name)

            self.training_management_service.registration_function(
                TrainingManagementServicerImpl(), self.server
            )

        # Add model runtime servicer to the gRPC server
        model_runtime_pb2_grpc.add_ModelRuntimeServicer_to_server(
            ModelRuntimeServicerImpl(interrupter=self.interrupter), self.server
        )
        service_names.append(
            model_runtime_pb2.DESCRIPTOR.services_by_name["ModelRuntime"].full_name
        )

        # Add runtime info servicer to the gRPC server
        runtime_info_service: ServicePackage = (
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.INFO,
            )
        )
        service_names.append(runtime_info_service.descriptor.full_name)

        runtime_info_service.registration_function(InfoServicer(), self.server)

        # Add gRPC default health servicer.
        # We use the non-blocking implementation to avoid thread starvation.
        health_servicer = health.HealthServicer(
            experimental_non_blocking=True,
            experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=1),
        )
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.server)

        # Finally enable service reflection after all services are added
        reflection.enable_server_reflection(service_names, self.server)

        # Listen on a unix socket as well for model mesh.
        if self.config.runtime.grpc.unix_socket_path and os.path.exists(
            os.path.dirname(self.config.runtime.grpc.unix_socket_path)
        ):
            try:
                self.server.add_insecure_port(
                    f"unix://{self.config.runtime.grpc.unix_socket_path}"
                )
                log.info(
                    "<RUN10001011I>",
                    "Caikit Runtime is communicating through address: unix://%s",
                    self.config.runtime.grpc.unix_socket_path,
                )
            except RuntimeError:
                log.info(
                    "<RUN10001100I>",
                    "Binding failed for: unix://%s",
                    self.config.runtime.grpc.unix_socket_path,
                )

        if (
            self.tls_config
            and self.tls_config.server.key
            and self.tls_config.server.cert
        ):
            log.info("<RUN10001805I>", "Running with TLS")

            tls_server_pair = (
                bytes(self._load_secret(self.tls_config.server.key), "utf-8"),
                bytes(self._load_secret(self.tls_config.server.cert), "utf-8"),
            )
            if self.tls_config.client.cert:
                log.info("<RUN10001806I>", "Running with mutual TLS")
                # Combine the client cert with the server's own cert so that
                # health probes can use the server's key/cert instead of needing
                # one signed by a potentially-external CA.
                root_certificates = b"\n".join(
                    [
                        bytes(self._load_secret(self.tls_config.client.cert), "utf-8"),
                        tls_server_pair[1],
                    ]
                )
                # Client will verify the server using server cert and the server
                # will verify the client using client cert.
                server_credentials = grpc.ssl_server_credentials(
                    [tls_server_pair],
                    root_certificates=root_certificates,
                    require_client_auth=True,
                )
            else:
                server_credentials = grpc.ssl_server_credentials([tls_server_pair])

            self.server.add_secure_port(f"[::]:{self.port}", server_credentials)
        else:
            log.info("<RUN10001807I>", "Running in insecure mode")
            self.server.add_insecure_port(f"[::]:{self.port}")

    def start(self, blocking: bool = True):
        """Boot the gRPC server. Can be non-blocking, or block until shutdown

        Args:
            blocking (boolean): Whether to block until shutdown
        """
        # Boot the thread interrupter
        if self.interrupter:
            self.interrupter.start()

        # Start the server. This is non-blocking, so we need to wait after
        self.server.start()

        log.info(
            "<RUN10001001I>",
            "Caikit Runtime is serving grpc on port: %s with thread pool size: %s",
            self.port,
            self.thread_pool._max_workers,
        )

        if blocking:
            self.server.wait_for_termination(None)

    def stop(self, grace_period_seconds: Optional[Union[float, int]] = None):
        """Stop the server, with an optional grace period.

        Args:
            grace_period_seconds (Union[float, int]): Grace period for service shutdown.
                Defaults to application config
        """
        log.info("Shutting down gRPC server")
        if grace_period_seconds is None:
            grace_period_seconds = (
                self.config.runtime.grpc.server_shutdown_grace_period_seconds
            )
        log.debug4("Stopping grpc server with %s grace seconds", grace_period_seconds)
        self.server.stop(grace_period_seconds)
        # Ensure we flush out any remaining billing metrics and stop metering
        if self.config.runtime.metering.enabled and self._global_predict_servicer:
            self._global_predict_servicer.stop_metering()
        # Shut down the model manager's model polling if enabled
        self._shut_down_model_manager()
        # Shut down the thread interrupter
        if self.interrupter:
            self.interrupter.stop()

    def render_protos(self, proto_out_dir: str) -> None:
        """Renders all the necessary protos for this service into a directory
        Args:
            proto_out_dir (str): Path to the directory to write proto files to
        """
        # First render out all the `@dataobject`s that should comprise the input / output
        # messages for these services
        caikit.core.data_model.render_dataobject_protos(proto_out_dir)

        # Then render each service
        if self.inference_service:
            self.inference_service.service.write_proto_file(proto_out_dir)
        if self.training_service:
            self.training_service.service.write_proto_file(proto_out_dir)

    def make_local_channel(self) -> grpc.Channel:
        """Return an insecure grpc channel over localhost for this server.
        Useful for unit testing or running local inference.
        """
        return grpc.insecure_channel(f"localhost:{self.port}")

    @staticmethod
    def _load_secret(secret: str) -> str:
        """If the secret points to a file, return the contents (plaintext reads).
        Else return the string"""
        if os.path.exists(secret):
            with open(secret, encoding="utf-8") as secret_file:
                return secret_file.read()
        return secret

    # Context manager impl
    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, type_, value, traceback):
        self.stop(0)


def main(blocking: bool = True):
    server = RuntimeGRPCServer()
    server.start(blocking)


if __name__ == "__main__":
    main()
