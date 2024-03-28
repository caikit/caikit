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
from typing import Dict, List, Optional, Union
import argparse
import json
import os
import shutil
import sys

# Third Party
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pb2, descriptor_pool

# First Party
from py_to_proto import descriptor_to_file
from py_to_proto.utils import safe_add_fd_to_pool
import alog

# Local
from ..config.config import get_config
from ..core.data_model import render_dataobject_protos
from ..core.data_model.dataobject import get_generated_proto_classes
from ..core.exceptions import error_handler
from .service_factory import ServicePackage, ServicePackageFactory
import caikit

log = alog.use_channel("RUNTIME-DUMP-SVC")
error = error_handler.get(log)

## Public ######################################################################


def dump_grpc_services(
    output_dir: str,
    write_modules_file: bool,
    consolidate: bool = False,
):
    """Utility for rendering the all generated interfaces to proto files

    Args:
        output_dir (str): The directory where the generated services should be
            placed
        write_modules_file (bool): Whether or not to write out the compatibility
            file for supported modules
        consolidate (bool): Whether or not to consolidate the generated protos
            by package
    """
    service_packages = _get_grpc_service_packages()
    if not consolidate:
        log.info(
            "Dumping raw service and data model protos without package consolidation"
        )
        render_dataobject_protos(output_dir)
        for svc_pkg in service_packages:
            svc_pkg.service.write_proto_file(output_dir)
    else:
        log.info("Dumping service and data model protos with package consolidation")
        os.makedirs(output_dir, exist_ok=True)
        all_descriptors = [
            proto_cls.DESCRIPTOR
            for proto_cls in get_generated_proto_classes()
            if proto_cls.DESCRIPTOR.file.pool is descriptor_pool.Default()
        ] + [pkg.descriptor for pkg in service_packages]
        fd_protos = _get_proto_file_descriptors(all_descriptors)
        _dump_consolidated_protos(fd_protos, output_dir)


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


## Implementation Details ######################################################


def _try_find_file_by_name(
    name: str,
    pool: descriptor_pool.DescriptorPool,
) -> Optional[_descriptor.FileDescriptor]:
    """Attempt to find a file descriptor by name and return None if not found"""
    try:
        return pool.FindFileByName(name)
    except KeyError:
        return None


def _recursive_safe_add_to_pool(
    fd_proto: descriptor_pb2.FileDescriptorProto,
    fd_protos_to_add: Dict[str, descriptor_pb2.FileDescriptorProto],
    dpool: descriptor_pool.DescriptorPool,
) -> _descriptor.FileDescriptor:
    """Recursively add the given file descriptor and all of its dependencies to
    the pool and handle double-add conflicts.
    """
    fds_to_add_by_file_name = {fd.name: fd for fd in fd_protos_to_add.values()}
    for dep_name in fd_proto.dependency:
        if not _try_find_file_by_name(dep_name, dpool):
            # Look in the pile of protos that need to be added
            if pending_fd_proto := fds_to_add_by_file_name.get(dep_name):
                _recursive_safe_add_to_pool(pending_fd_proto, fd_protos_to_add, dpool)
            # Look in the default pool
            elif dflt_fd := _try_find_file_by_name(dep_name, descriptor_pool.Default()):
                dep_fd_proto = descriptor_pb2.FileDescriptorProto()
                dflt_fd.CopyToProto(dep_fd_proto)
                _recursive_safe_add_to_pool(dep_fd_proto, fd_protos_to_add, dpool)
            else:
                error(
                    "<COR25660790E>",
                    ValueError(
                        f"Can't add {fd_proto.name}: dependency {dep_name} not found"
                    ),
                )
    safe_add_fd_to_pool(fd_proto, dpool)
    return dpool.FindFileByName(fd_proto.name)


def _descriptor_to_proto(
    descriptor: Union[
        _descriptor.Descriptor,
        _descriptor.EnumDescriptor,
        _descriptor.ServiceDescriptor,
    ],
) -> Union[
    descriptor_pb2.DescriptorProto,
    descriptor_pb2.EnumDescriptorProto,
    descriptor_pb2.ServiceDescriptorProto,
]:
    """Convert a given Descriptor type to the corresponding Proto for
    comparison by content rather than instance id
    """
    error.type_check(
        "<COR46719006E>",
        _descriptor.Descriptor,
        _descriptor.EnumDescriptor,
        _descriptor.ServiceDescriptor,
        descriptor=descriptor,
    )
    proto_type = None
    if isinstance(descriptor, _descriptor.Descriptor):
        proto_type = descriptor_pb2.DescriptorProto
    elif isinstance(descriptor, _descriptor.EnumDescriptor):
        proto_type = descriptor_pb2.EnumDescriptorProto
    elif isinstance(descriptor, _descriptor.ServiceDescriptor):
        proto_type = descriptor_pb2.ServiceDescriptorProto
    assert proto_type
    proto = proto_type()
    descriptor.CopyToProto(proto)
    return proto


def _get_proto_file_descriptors(
    object_descriptors: List[
        Union[
            _descriptor.Descriptor,
            _descriptor.EnumDescriptor,
            _descriptor.ServiceDescriptor,
        ]
    ],
) -> Dict[str, descriptor_pb2.FileDescriptorProto]:
    """Get a dict mapping package names to consolidated DescriptorProto objects
    holding all auto-generated messages and enums in the given package.
    """

    # Deduplicate object descriptors
    dup_candidates = {}
    for obj_desc in object_descriptors:
        dup_candidates.setdefault(f"{type(obj_desc)}/{obj_desc.full_name}", {})[
            id(obj_desc)
        ] = obj_desc
    dups = {
        dup_name: obj_descs
        for dup_name, obj_descs in dup_candidates.items()
        if len(
            {
                _descriptor_to_proto(obj_desc).SerializeToString()
                for obj_desc in obj_descs.values()
            }
        )
        > 1
    }
    error.value_check(
        "<COR01018988E>",
        not dups,
        "Found conflicting definitions of protobuf objects: {}",
        list(dups.keys()),
    )
    object_descriptors = sorted(
        [list(obj_descs.values())[0] for obj_descs in dup_candidates.values()],
        key=lambda obj_desc: obj_desc.name,
    )

    # Collect the auto-gen protos by package
    file_descriptor_protos = {}
    for obj_desc in object_descriptors:
        file_descriptor_proto = file_descriptor_protos.setdefault(
            obj_desc.file.package, descriptor_pb2.FileDescriptorProto()
        )
        obj_desc.file.CopyToProto(file_descriptor_proto)

    # Update the file names to be package-level
    for pkg_name, pkg_fd in file_descriptor_protos.items():
        file_safe_pkg_name = pkg_name.replace(".", "_")
        pkg_fd.name = f"{file_safe_pkg_name}.proto"

    # Update the dependencies for each package-level file descriptor proto
    for pkg_name, pkg_fd in file_descriptor_protos.items():

        # Figure out the remaining set of deps for this file as all external
        # deps and all generated package-level files that aren't this one
        pkg_deps = set()
        for candidate_pkg_name in file_descriptor_protos:
            if candidate_pkg_name != pkg_name and any(
                dep.startswith(candidate_pkg_name) for dep in pkg_fd.dependency
            ):
                pkg_deps.add(candidate_pkg_name)

        # Clear out existing object-level file deps
        for existing_dep in list(pkg_fd.dependency):
            if any(
                existing_dep.startswith(candidate_pkg_name)
                for candidate_pkg_name in file_descriptor_protos
            ):
                pkg_fd.dependency.remove(existing_dep)

        # Add package-level dependency files
        pkg_fd.dependency.extend(
            sorted([file_descriptor_protos[pkg].name for pkg in pkg_deps])
        )

    return file_descriptor_protos


def _dump_consolidated_protos(
    fd_protos: Dict[str, descriptor_pb2.FileDescriptorProto],
    interfaces_dir: str,
):
    """Dump all protobuf interfaces consolidated by package"""
    temp_dpool = descriptor_pool.DescriptorPool()
    for fd_proto in fd_protos.values():
        fd = _recursive_safe_add_to_pool(fd_proto, fd_protos, temp_dpool)
        target_file = os.path.join(interfaces_dir, fd.name)
        with open(target_file, "w") as handle:
            handle.write(descriptor_to_file(fd))


def _get_grpc_service_packages(
    write_modules_file: bool = False,
) -> List[ServicePackage]:
    """Get all enabled grpc service packages"""
    inf_enabled = get_config().runtime.service_generation.enable_inference
    train_enabled = get_config().runtime.service_generation.enable_training
    svc_descriptors = []
    if inf_enabled:
        svc_descriptors.append(
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.INFERENCE,
                write_modules_file=write_modules_file,
            )
        )
    if train_enabled:
        svc_descriptors.append(
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.TRAINING,
            )
        )
        svc_descriptors.append(
            ServicePackageFactory.get_service_package(
                ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
            )
        )
    svc_descriptors.append(
        ServicePackageFactory.get_service_package(
            ServicePackageFactory.ServiceType.INFO,
        )
    )
    return svc_descriptors


## Main ########################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Dump grpc and http services for inference and train"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory for service(s)' proto files",
    )
    parser.add_argument(
        "-j",
        "--write-modules-json",
        default=False,
        action="store_true",
        help="Wether the modules.json (of supported modules) should be output?",
    )
    parser.add_argument(
        "-c",
        "--clean",
        default=False,
        action="store_true",
        help="Clean out existing content in output dir",
    )
    parser.add_argument(
        "-p",
        "--consolidate-packages",
        default=False,
        action="store_true",
        help="Consolidate protobufs by package",
    )
    args = parser.parse_args()

    # Set up logging so users can set LOG_LEVEL etc
    caikit.core.toolkit.logging.configure()

    # Make sure the output dir exists and optionally clean it out
    out_dir = args.output_dir
    if args.clean and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if get_config().runtime.grpc.enabled:
        dump_grpc_services(
            out_dir,
            args.write_modules_json,
            args.consolidate_packages,
        )
    if get_config().runtime.http.enabled:
        dump_http_services(out_dir)


if __name__ == "__main__":
    main()
