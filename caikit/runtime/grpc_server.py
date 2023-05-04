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
import signal
import socket

# Third Party
from grpc_health.v1 import health, health_pb2_grpc
from prometheus_client import start_http_server
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
from caikit.runtime.protobufs import model_runtime_pb2_grpc, process_pb2_grpc
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.servicers.model_runtime_servicer import ModelRuntimeServicerImpl
from caikit.runtime.servicers.model_train_servicer import ModelTrainServicerImpl
from caikit.runtime.servicers.training_management_servicer import (
    TrainingManagementServicerImpl,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

# Have pylint ignore broad exception catching in this file so that we can log all
# unexpected errors using alog.
# pylint: disable=W0703
import caikit.core.data_model

log = alog.use_channel("GRPC-SERVR")
PROMETHEUS_METRICS_INTERCEPTOR = PromServerInterceptor()


class RuntimeGRPCServer:
    """An implementation of a gRPC server that serves caikit runtimes"""

    def __init__(
        self,
        inference_service: ServicePackage,
        training_service: Optional[ServicePackage],
        tls_config_override: aconfig.Config = None,
        handle_terminations: bool = False,
    ):
        self.config = get_config()
        self.inference_service = inference_service
        self.training_service = training_service

        self.port = (
            self._find_port(self.config.runtime.port)
            if self.config.runtime.find_available_port
            else self.config.runtime.port
        )
        if self.port != self.config.runtime.port:
            log.warning(
                "Port %s was in use, had to find another!", self.config.runtime.port
            )

        # Initialize basic server
        # py_grpc_prometheus.server_metrics.
        self.server = grpc.server(
            futures.ThreadPoolExecutor(
                max_workers=self.config.runtime.server_thread_pool_size
            ),
            interceptors=(PROMETHEUS_METRICS_INTERCEPTOR,),
        )

        # Intercept an Inference Service
        self._global_predict_servicer = GlobalPredictServicer(self.inference_service)
        self.server = CaikitRuntimeServerWrapper(
            server=self.server,
            global_predict=self._global_predict_servicer.Predict,
            intercepted_svc_descriptor=self.inference_service.descriptor,
            excluded_methods=["/natural_language_understanding.CaikitRuntime/Echo"],
        )

        # Register inference service
        self.inference_service.registration_function(
            self.inference_service.service, self.server
        )

        # And intercept a training service, if we have one
        if self.training_service:
            global_train_servicer = GlobalTrainServicer(self.training_service)
            self.server = CaikitRuntimeServerWrapper(
                server=self.server,
                global_predict=global_train_servicer.Train,
                intercepted_svc_descriptor=self.training_service.descriptor,
                excluded_methods=None,
            )

            # Register training service
            self.training_service.registration_function(
                self.training_service.service, self.server
            )

            # Add model train servicer to the gRPC server
            process_pb2_grpc.add_ProcessServicer_to_server(
                ModelTrainServicerImpl(self.training_service), self.server
            )

            # Add training management servicer to the gRPC server
            training_management_service: ServicePackage = (
                ServicePackageFactory.get_service_package(
                    ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
                    ServicePackageFactory.ServiceSource.GENERATED,
                )
            )

            training_management_service.registration_function(
                TrainingManagementServicerImpl(), self.server
            )

        # Add model runtime servicer to the gRPC server
        model_runtime_pb2_grpc.add_ModelRuntimeServicer_to_server(
            ModelRuntimeServicerImpl(), self.server
        )

        # Add gRPC default health servicer.
        # We use the non-blocking implementation to avoid thread starvation.
        health_servicer = health.HealthServicer(
            experimental_non_blocking=True,
            experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=1),
        )
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.server)

        # Listen on a unix socket as well for model mesh.
        if self.config.runtime.unix_socket_path and os.path.exists(
            self.config.runtime.unix_socket_path
        ):
            try:
                self.server.add_insecure_port(
                    f"unix://{self.config.runtime.unix_socket_path}"
                )
                log.info(
                    "<RUN10001011I>",
                    "Caikit Runtime is communicating through address: unix://%s",
                    self.config.runtime.unix_socket_path,
                )
            except RuntimeError:
                log.info(
                    "<RUN10001100I>",
                    "Binding failed for: unix://%s",
                    self.config.runtime.unix_socket_path,
                )

        # Pull TLS config from app config, unless an explicit override was passed
        self.tls_config = (
            tls_config_override if tls_config_override else self.config.runtime.tls
        )

        if (
            self.tls_config
            and self.tls_config.server.key
            and self.tls_config.server.cert
        ):
            log.info("<RUN10001805I>", "Running with TLS")

            tls_server_pair = (
                bytes(load_secret(self.tls_config.server.key), "utf-8"),
                bytes(load_secret(self.tls_config.server.cert), "utf-8"),
            )
            if self.tls_config.client.cert:
                log.info("<RUN10001806I>", "Running with mutual TLS")
                # Client will verify the server using server cert and the server
                # will verify the client using client cert.
                server_credentials = grpc.ssl_server_credentials(
                    [tls_server_pair],
                    root_certificates=load_secret(self.tls_config.client.cert),
                    require_client_auth=True,
                )
            else:
                server_credentials = grpc.ssl_server_credentials([tls_server_pair])

            self.server.add_secure_port(f"[::]:{self.port}", server_credentials)
        else:
            log.info("<RUN10001807I>", "Running in insecure mode")
            self.server.add_insecure_port(f"[::]:{self.port}")

        # NOTE: signal function can only be called from main thread of the main interpreter.
        # If this function is called from a thread (like in tests) then signal handler cannot
        # be used. Thus, we will only have real termination_handler when this is called from
        # the __main__.
        if handle_terminations:
            signal.signal(signal.SIGINT, self.interrupt)
            signal.signal(signal.SIGTERM, self.interrupt)

    def start(self, blocking: bool = True):
        """Boot the gRPC server. Can be non-blocking, or block until shutdown

        Args:
            blocking (boolean): Whether to block until shutdown
        """
        # Start the server. This is non-blocking, so we need to wait after
        self.server.start()

        log.info(
            "<RUN10001001I>",
            "Caikit Runtime is serving on port: %s with thread pool size: %s",
            self.port,
            self.config.runtime.server_thread_pool_size,
        )

        if blocking:
            self.server.wait_for_termination(None)

    def interrupt(self, signal_, _stack_frame):
        """Interrupt the server. Implements the interface for Python signal.signal

        Reference: https://docs.python.org/3/library/signal.html#signal.signal
        """
        log.info(
            "<RUN81000120I>",
            "Caikit Runtime received interrupt signal %s, shutting down",
            signal_,
        )
        self.stop()

    def stop(self, grace_period_seconds: Union[float, int] = None):
        """Stop the server, with an optional grace period.

        Args:
            grace_period_seconds (Union[float, int]): Grace period for service shutdown.
                Defaults to application config
        """
        if not grace_period_seconds:
            grace_period_seconds = (
                self.config.runtime.server_shutdown_grace_period_seconds
            )
        self.server.stop(grace_period_seconds)
        # Ensure we flush out any remaining billing metrics and stop metering
        if self.config.runtime.metering.enabled:
            self._global_predict_servicer.rpc_meter.flush_metrics()
            self._global_predict_servicer.rpc_meter.end_writer_thread()

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

    @classmethod
    def _find_port(cls, start=8888, end=None, host="127.0.0.1"):
        """Function to find an available port in a given range
        Args:
            start: int
                Starting number for port search (inclusive)
                Default: 8888
            end: Optional[int]
                End number for port search (exclusive)
                Default: start + 1000
            host: str
                Host name or ip address to search on
                Default: localhost
        Returns:
            int
                Available port
        """
        end = start + 1000 if end is None else end
        if start < end:
            with socket.socket() as soc:
                # soc.connect_ex returns 0 if connection is successful and thus
                # indicating port is available
                if soc.connect_ex((host, start)) == 0:
                    # port is in use, thus connection to it is successful
                    return cls._find_port(start + 1, end, host)

                # port is open
                return start

    # Context manager impl
    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, type_, value, traceback):
        self.stop(0)


def load_secret(secret: str) -> str:
    """If the secret points to a file, return the contents (plaintext reads).
    Else return the string"""
    if os.path.exists(secret):
        with open(secret, "r", encoding="utf-8") as secret_file:
            return secret_file.read()
    return secret


def main():
    # Configure using the log level and formatter type specified in config.
    caikit.core.toolkit.logging.configure()

    # Start serving Prometheus metrics
    caikit_config = get_config()
    start_http_server(caikit_config.runtime.metrics.port)

    # Enable signal handling
    handle_terminations = True

    # Assume we want compiled services for now
    service_source = (
        ServicePackageFactory.ServiceSource.GENERATED
        if caikit_config.runtime.service_generation.enabled
        else ServicePackageFactory.ServiceSource.COMPILED
    )
    log.debug("Running with service source: %s", service_source)

    # We should always be able to stand up an inference service
    inference_service: ServicePackage = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
        service_source,
    )

    # But maybe not always a training service
    try:
        training_service: Optional[
            ServicePackage
        ] = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING,
            service_source,
        )
    except CaikitRuntimeException as e:
        log.warning("Cannot stand up training service, disabling training: %s", e)
        training_service = None

    server = RuntimeGRPCServer(
        inference_service=inference_service,
        training_service=training_service,
        handle_terminations=handle_terminations,
    )
    server.start()


if __name__ == "__main__":
    main()
