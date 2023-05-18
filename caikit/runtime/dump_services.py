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
import sys

# Local
from ..core.data_model import render_dataobject_protos
from .service_factory import ServicePackageFactory
import caikit


def dump_services(output_dir: str):
    """
    Utility for rendering the all generated interfaces to proto files
    """
    inf_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.INFERENCE,
        ServicePackageFactory.ServiceSource.GENERATED,
    )
    train_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING,
        ServicePackageFactory.ServiceSource.GENERATED,
    )
    train_mgt_svc = ServicePackageFactory.get_service_package(
        ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
        ServicePackageFactory.ServiceSource.GENERATED,
    )

    render_dataobject_protos(output_dir)
    inf_svc.service.write_proto_file(output_dir)
    train_svc.service.write_proto_file(output_dir)
    train_mgt_svc.service.write_proto_file(output_dir)


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <output_dir>"
    out_dir = sys.argv[1]
    # Set up logging so users can set LOG_LEVEL etc
    caikit.core.toolkit.logging.configure()

    dump_services(out_dir)
