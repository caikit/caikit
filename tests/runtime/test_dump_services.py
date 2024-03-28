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
import os
import shutil
import tempfile

# Third Party
import pytest

# First Party
import alog

# Local
from caikit.runtime.dump_services import dump_grpc_services, dump_http_services
from tests.conftest import ARM_ARCH, PROTOBUF_VERSION

## Helpers #####################################################################

log = alog.use_channel("TEST-DUMP-I")


@pytest.mark.skipif(
    PROTOBUF_VERSION < 4 and ARM_ARCH, reason="protobuf 3 serialization bug"
)
def test_dump_grpc_services_dir_exists():
    with tempfile.TemporaryDirectory() as workdir:
        dump_grpc_services(workdir, False)
        assert os.path.exists(workdir)

        for file in os.listdir(workdir):
            assert file.endswith(".proto")


@pytest.mark.skipif(
    PROTOBUF_VERSION < 4 and ARM_ARCH, reason="protobuf 3 serialization bug"
)
def test_dump_grpc_services_dir_does_not_exist():
    with tempfile.TemporaryDirectory() as workdir:
        fake_dir = os.path.join(workdir, "fake_dir")
        dump_grpc_services(fake_dir, False)
        assert os.path.exists(fake_dir)

        for file in os.listdir(fake_dir):
            assert file.endswith(".proto")

        shutil.rmtree(fake_dir)


@pytest.mark.skipif(
    PROTOBUF_VERSION < 4 and ARM_ARCH, reason="protobuf 3 serialization bug"
)
def test_dump_grpc_services_consolidated():
    with tempfile.TemporaryDirectory() as workdir:
        dump_grpc_services(workdir, False, consolidate=True)
        assert os.path.exists(workdir)
        # Make sure the file names match the expected names for caikit plus
        # sample_lib
        # NOTE: Dumping services dumps _all_ data model objects, so we cannot
        #   do an exact check due to the global descriptor pool and other tests
        #   that modify it.
        dumped_files = os.listdir(workdir)
        exp_fnames = {
            "caikit_runtime_SampleLib.proto",
            "caikit_runtime_info.proto",
            "caikit_runtime_training.proto",
            "caikit_data_model_common.proto",
            "caikit_data_model_common_runtime.proto",
            "caikit_data_model_runtime.proto",
            "caikit_data_model_sample_lib.proto",
        }
        assert all(fname in dumped_files for fname in exp_fnames)

        # Spot check one of the files that we know will have specific contents
        with open(os.path.join(workdir, "caikit_runtime_info.proto")) as handle:
            content = handle.read()
            assert "package caikit.runtime.info;" in content
            assert "service InfoService" in content
            assert "rpc GetRuntimeInfo" in content


def test_dump_http_services_dir_exists():
    with tempfile.TemporaryDirectory() as workdir:
        dump_http_services(workdir)
        assert os.path.exists(workdir)

        for file in os.listdir(workdir):
            assert file == "openapi.json"
            assert os.path.getsize(os.path.join(workdir, file)) > 0


def test_dump_http_services_dir_does_not_exist():
    with tempfile.TemporaryDirectory() as workdir:
        fake_dir = os.path.join(workdir, "fake_dir")
        dump_http_services(fake_dir)
        assert os.path.exists(fake_dir)

        for file in os.listdir(fake_dir):
            assert file == "openapi.json"
            assert os.path.getsize(os.path.join(fake_dir, file)) > 0
