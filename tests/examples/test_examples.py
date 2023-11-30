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
from os import getenv, path
from tempfile import TemporaryDirectory
import asyncio
import subprocess

# Third Party
import pytest

# Local
from caikit.config.config import get_config
from caikit.runtime.__main__ import main
from caikit.runtime.dump_services import dump_grpc_services, dump_http_services
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.examples.shared import requirements, waitForPort
import caikit


@pytest.mark.examples
def test_example_text_sentiment():
    # Example specific grpc port
    grpc_port = 8085

    with requirements("text-sentiment") as (python_venv, example_dir):
        # Start the server
        with subprocess.Popen(
            [python_venv, "start_runtime.py"],
            cwd=example_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as server:
            # Check if the gRPC port is open
            # The gRPC server start-up time has some inherent variability
            # 60s timeout should cover most situations, while keeping the
            # test execution time reasonable
            if not asyncio.run(waitForPort(grpc_port, 60)):
                server.terminate()
                pytest.fail(
                    "Failed to connect to the gRPC server on port {} in 30s.".format(
                        grpc_port
                    )
                )

            # Server is running, start the client
            # Use a timeout of 10s for inference. Capture outputs to report
            # them in case of failure.
            try:
                subprocess.run(
                    [python_venv, path.join(example_dir, "client.py")],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                server.terminate()
                pytest.fail("Client failed with output: {}".format(e))

            # Client worked well, let's stop the server
            server.terminate()


def test_lazy_load_local_models_invalid_model_dire():
    """Make sure an ephemeral model (not on disk) can be lazy loaded if the
    right finder configuration is present to load it without hitting disk.
    """

    # breakpoint()

    invalid_local_models_dir = path.abspath(
        path.join(path.dirname(__file__), "invalid")
    )
    workdir = path.abspath(path.join(path.dirname(__file__), "models"))

    with TemporaryDirectory() as workdir:
        print("BANG1")

        # with pytest.raises(CaikitRuntimeException) as context:
        # with pytest.raises(Exception) as e_info:
        caikit.config.configure(
            config_dict={
                "merge_strategy": "merge",
                "runtime": {
                    "library": "sample_lib",
                    "local_models_dir": invalid_local_models_dir,
                    "lazy_load_local_models": True,
                    "grpc": {"enabled": True},
                    "http": {"enabled": True},
                    "training": {"save_with_id": False, "output_dir": workdir},
                    "service_generation": {
                        "package": "caikit_sample_lib"
                    },  # This is done to avoid name collision with Caikit itself
                },
            }
        )

        # print(f"Message: {context.value.status_code}")
        # print(e_info.type)  # This will output the type of the exception
        # print(e_info.value)  # This will output the message of the exception

    print("BANG2")
