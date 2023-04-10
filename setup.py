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

"""
Setup to be able to build caikit library.
"""
# Standard
import glob
import os

# Third Party
import setuptools.command.build_py

# get version of library
CAIKIT_VERSION = os.getenv("CAIKIT_VERSION")
if not CAIKIT_VERSION:
    raise RuntimeError("CAIKIT_VERSION must be set")

# base directory containing caikit (location of this file)
base_dir = os.path.dirname(os.path.realpath(__file__))

# read requirements from file
with open(os.path.join(base_dir, "requirements.txt"), encoding="utf-8") as filehandle:
    requirements = filehandle.read().splitlines()

setuptools.setup(
    name="caikit",
    author="caikit",
    version=CAIKIT_VERSION,
    python_requires=">=3.8",
    license="Copyright Caikit Authors 2023 -- All rights reserved.",
    description="AI toolkit that enables AI users to consume stable task-specific "
    "model APIs and enables AI developers build algorithms and models in a "
    "modular/composable framework",
    install_requires=requirements,
    packages=setuptools.find_packages(include=("caikit*",)),
    data_files=[
        os.path.join("caikit", "core", "config", "config.yml"),
        os.path.join("caikit", "runtime", "config", "config.yml"),
    ]
    + glob.glob(os.path.join("caikit", "runtime", "protobufs", "protos", "*.proto")),
    include_package_data=True,
)
