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
from os import path
import argparse
import sys

# First Party
import alog

# Local
from caikit.runtime import grpc_server, http_server
import caikit


def protocol_arg():
    parser = argparse.ArgumentParser(description="protocol switch")
    parser.add_argument(
        "--protocol", type=str, default="grpc", help="Specify a protocol: grpc or http"
    )

    args = parser.parse_args()
    print(f"The specified protocol is: {args.protocol}")
    return args.protocol


if __name__ == "__main__":
    models_directory = path.abspath(path.join(path.dirname(__file__), "models"))
    caikit.config.configure(
        config_dict={
            "merge_strategy": "merge",
            "runtime": {
                "local_models_dir": models_directory,
                "library": "text_sentiment",
            },
        }
    )

    sys.path.append(
        path.abspath(path.join(path.dirname(__file__), "../"))
    )  # Here we assume that `start_runtime` file is at the same level of the
    # `text_sentiment` package

    alog.configure(default_level="debug")

    protocol = protocol_arg()
    if protocol == "grpc":
        grpc_server.main()
    elif protocol == "http":
        http_server.main()
    else:
        print("--protocol must be one of [grpc, http]")
