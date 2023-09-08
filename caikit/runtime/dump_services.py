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
import json
import os
import sys

# First Party
import alog

# Local
from ..core.data_model import render_dataobject_protos
from .service_factory import ServicePackageFactory
from caikit.config.config import get_config
import caikit

log = alog.use_channel("RUNTIME-DUMP-SVC")


def dump_grpc_services(output_dir: str):
    """Utility for rendering the all generated interfaces to proto files"""
    inf_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
    )
    train_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
    )
    train_mgt_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
    )
    render_dataobject_protos(output_dir)
    inf_svc.service.write_proto_file(output_dir)
    train_svc.service.write_proto_file(output_dir)
    train_mgt_svc.service.write_proto_file(output_dir)


def dump_http_services(output_dir: str):
    """Dump out the openapi.json for the HTTP server"""

    # Import the HTTP components inside the dump function to avoid requiring
    # them when dumping grpc interfaces without the `runtime-http` optional
    # dependencies installed.

    try:
        # Third Party
        from fastapi.testclient import (  # pylint: disable=import-outside-toplevel
            TestClient,
        )

        # Local
        from .http_server import (  # pylint: disable=import-outside-toplevel
            RuntimeHTTPServer,
        )
    except ModuleNotFoundError as e:
        message = (
            "Error: {} - unable to dump http services. Perhaps you missed"
            " installing the http optional dependencies?".format(e)
        )
        log.error("<DMP76165827E>", message)
        sys.exit(1)

    server = RuntimeHTTPServer()
    with TestClient(server.app) as client:
        response = client.get("/openapi.json")
        response.raise_for_status()
        with open(
            os.path.join(output_dir, "openapi.json"), "w", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <output_dir>"
    out_dir = sys.argv[1]
    # Set up logging so users can set LOG_LEVEL etc
    caikit.core.toolkit.logging.configure()

    if get_config().runtime.grpc.enabled:
        dump_grpc_services(out_dir)
    if get_config().runtime.http.enabled:
        dump_http_services(out_dir)
