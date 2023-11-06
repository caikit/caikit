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
import argparse
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


def dump_grpc_services(output_dir: str, write_modules_file):
    """Utility for rendering the all generated interfaces to proto files"""
    inf_enabled = get_config().runtime.service_generation.enable_inference
    train_enabled = get_config().runtime.service_generation.enable_training

    if inf_enabled:
        inf_svc = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE,
            write_modules_file=write_modules_file,
        )
    if train_enabled:
        train_svc = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING,
        )
        train_mgt_svc = ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
        )
    info_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFO,
    )

    render_dataobject_protos(output_dir)
    if inf_enabled:
        inf_svc.service.write_proto_file(output_dir)
    if train_enabled:
        train_svc.service.write_proto_file(output_dir)
        train_mgt_svc.service.write_proto_file(output_dir)
    info_svc.service.write_proto_file(output_dir)


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

        # create output dir if doesn't exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        with open(
            os.path.join(output_dir, "openapi.json"), "w", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(response.json(), indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Dump grpc and http services for inference and train"
    )

    # Add an argument for the output_dir
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory for service(s)' proto files",
    )

    # Add an argument for write_modules_json
    parser.add_argument(
        "-j",
        "--write-modules-json",
        default=False,
        action="store_true",
        help="Wether the modules.json (of supported modules) should be output?",
    )

    args = parser.parse_args()

    out_dir = args.output_dir
    write_modules_json = args.write_modules_json

    # Set up logging so users can set LOG_LEVEL etc
    caikit.core.toolkit.logging.configure()

    if get_config().runtime.grpc.enabled:
        dump_grpc_services(out_dir, write_modules_json)
    if get_config().runtime.http.enabled:
        dump_http_services(out_dir)


if __name__ == "__main__":
    main()
