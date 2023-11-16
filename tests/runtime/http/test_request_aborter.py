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
import asyncio
import datetime
import threading
import time

# Third Party
from fastapi import FastAPI, Request
from requests.exceptions import ReadTimeout
import pytest
import requests
import uvicorn

# Local
from caikit.runtime.http.request_aborter import HttpRequestAborter


def get_time_remaining(start_time: datetime.datetime, timeout: int = 10) -> int:
    now = datetime.datetime.now()
    return ((start_time + datetime.timedelta(seconds=timeout)) - now).seconds


def test_request_aborter(open_port):
    # Get start time to ensure test doesn't hang
    start_time = datetime.datetime.now()

    # Create FastAPI app for testing
    app = FastAPI()

    # Initialize synchronization variables for tracking request process
    abort_event = threading.Event()
    request_finished = threading.Event()

    # Define an endpoint that sleeps until the client disconnects.
    @app.get("/test_abort")
    async def test_aborter(context: Request):
        # Create aborter and add parent event
        TEST_ABORTER = HttpRequestAborter(context, poll_time=0.001)
        TEST_ABORTER.add_event(abort_event)

        # Wait for client to disconnect
        while not abort_event.is_set():
            await asyncio.sleep(0.001)

        # Assign TEST_ABORTER to the parent function. This allows the test to have
        # access to this object without using globals
        test_request_aborter.TEST_ABORTER = TEST_ABORTER
        request_finished.set()

    # Start up a local uvicorn server in a thread
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=open_port,
        log_level="trace",
        log_config=None,
        timeout_graceful_shutdown=None,
    )
    server = uvicorn.Server(config=config)
    server_thread = threading.Thread(target=server.run)
    server_thread.start()

    server_exception = None
    try:
        # Wait for uvicorn to start
        while not server.started:
            if get_time_remaining(start_time) < 0:
                raise TimeoutError("Server did not start in time")

            time.sleep(0.01)

        # Try the endpoint but timeout after 1 second
        with pytest.raises(ReadTimeout):
            requests.get(
                f"http://localhost:{open_port}/test_abort",
                timeout=0.1,
            )

        # Wait for the request to finish/abort
        request_finished.wait(get_time_remaining(start_time))

        # Assert the request aborter actually aborted
        assert test_request_aborter.TEST_ABORTER.must_abort()
        assert abort_event.is_set()
        assert request_finished.is_set()

    except Exception as exc:
        server_exception = exc
    finally:
        # Clean up the server
        server.should_exit = True
        server_thread.join()

    if server_exception:
        raise server_exception
