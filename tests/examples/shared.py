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

# Standard
from contextlib import contextmanager
from os import path
from typing import Tuple
import asyncio
import subprocess
import tempfile
import time
import venv

# Third Party
import pytest

caikit_dir = path.abspath(path.join(path.dirname(__file__), "..", ".."))


async def waitForPort(port, timeout: int) -> bool:
    tmax = time.time() + timeout
    while time.time() < tmax:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), timeout=5
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            await asyncio.sleep(1)
    return False


@contextmanager
def requirements(example: str) -> Tuple[str, str]:
    """Installs caikit and requirements.txt from relative path, returns python from venv
    and absolute path to the example.
    Raises subprocess.CalledProcessError if installation fails
    """
    with tempfile.TemporaryDirectory() as venv_dir:
        python = path.abspath(path.join(venv_dir, "bin", "python3"))
        pip = path.abspath(path.join(venv_dir, "bin", "pip3"))
        example_dir = path.abspath(path.join(caikit_dir, "examples", example))

        # Create a venv, install local version of caikit and local version of requirements
        venv.create(env_dir=venv_dir, system_site_packages=False, with_pip=True)
        try:
            subprocess.run([pip, "install", caikit_dir + "[all]"], check=True)
            subprocess.run(
                [
                    pip,
                    "install",
                    "-r",
                    path.join(example_dir, "requirements.txt"),
                ],
                check=True,
            )
            yield python, example_dir
        except subprocess.CalledProcessError as cpe:
            pytest.fail(
                "Could not install requirements for {}: {}".format(example, cpe)
            )
