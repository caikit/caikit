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
# First Party
import alog

# Local
from caikit.config.config import get_config

log = alog.use_channel("RUNTIME-MAIN")


def main():
    _grpc_server = None
    _http_server = None

    #####################
    # Start the servers
    #####################

    # Start serving grpc server
    if get_config().runtime.grpc.enabled:
        # Import the gRPC components inside the function to avoid requiring
        # them when starting runtime without the `runtime-grpc` optional
        # dependencies installed.

        try:
            # Local
            from caikit.runtime.grpc_server import (  # pylint: disable=import-outside-toplevel
                RuntimeGRPCServer,
            )
        except ModuleNotFoundError as e:
            message = (
                "Error: {} - unable to start gRPC server. Perhaps you missed"
                " installing the gRPC optional dependencies with"
                "`pip install caikit[runtime-grpc]`?".format(e)
            )
            log.error("<RUN72169927E>", message)
            raise

        log.debug("Starting up caikit.runtime.grpc_server")

        _grpc_server = RuntimeGRPCServer()
        _grpc_server.start(blocking=not get_config().runtime.http.enabled)

    # Start serving http server
    if get_config().runtime.http.enabled:
        # Import the HTTP components inside the function to avoid requiring
        # them when starting runtime without the `runtime-http` optional
        # dependencies installed.

        try:
            # Local
            from caikit.runtime.http_server import (  # pylint: disable=import-outside-toplevel
                RuntimeHTTPServer,
            )
        except ModuleNotFoundError as e:
            message = (
                "Error: {} - unable to start REST server. Perhaps you missed"
                " installing the http optional dependencies with"
                " `pip install caikit[runtime-http]`?".format(e)
            )
            log.error("<RUN76169927E>", message)
            raise

        log.debug("Starting up caikit.runtime.http_server")

        _http_server = RuntimeHTTPServer()
        _http_server.start(blocking=True)  # make http always blocking


if __name__ == "__main__":
    main()
