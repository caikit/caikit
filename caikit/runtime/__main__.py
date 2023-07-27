# Standard
from typing import Optional
import signal
import sys

# Third Party
from prometheus_client import start_http_server

# First Party
import alog

# Local
from caikit.config.config import get_config
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.http_server import RuntimeHTTPServer
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit

log = alog.use_channel("RUNTIME-MAIN")

# Make sample_lib available for import, useful for local testing
sys.path.append("tests/fixtures")


def interrupt(signal_, _stack_frame):
    log.info(
        "<RUN87630120I>",
        "Caikit Runtime received interrupt signal %s, shutting down",
        signal_,
    )
    grpc_server.stop()
    http_server.stop()


# NOTE: signal function can only be called from main thread of the main
# interpreter. If this function is called from a thread (like in tests)
# then signal handler cannot be used. Thus, we will only have real
# termination_handler when this is called from the __main__.

signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, interrupt)

# Configure using the log level and formatter type specified in config.
caikit.core.toolkit.logging.configure()

caikit_config = get_config()

# We should always be able to stand up an inference service
inference_service: ServicePackage = ServicePackageFactory.get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

# But maybe not always a training service
try:
    training_service: Optional[
        ServicePackage
    ] = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
    )
except CaikitRuntimeException as e:
    log.warning("Cannot stand up training service, disabling training: %s", e)
    training_service = None

#####################
# Start the servers
#####################

# Start serving Prometheus metrics
if caikit_config.runtime.metrics.enabled:
    log.info(
        "Serving prometheus metrics on port %s", caikit_config.runtime.metrics.port
    )
    with alog.ContextTimer(log.info, "Booted metrics server in "):
        start_http_server(caikit_config.runtime.metrics.port)


# Start serving http server
if caikit_config.runtime.http.enabled:
    log.debug("Starting up caikit.runtime.http_server")

    http_server = RuntimeHTTPServer(
        inference_service=inference_service, training_service=training_service
    )
    http_server.start(blocking=not caikit_config.runtime.grpc.enabled)

# Start serving grpc server
if caikit_config.runtime.grpc.enabled:
    log.debug("Starting up caikit.runtime.grpc_server")

    grpc_server = RuntimeGRPCServer(
        inference_service=inference_service,
        training_service=training_service,
    )
    grpc_server.start(blocking=True)
