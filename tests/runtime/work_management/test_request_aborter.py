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
from caikit.runtime.work_management.request_aborter import RequestAborter


def check_server_timeout(
    start_time: datetime.datetime, server: uvicorn.Server, timeout: int = 10
):
    now = datetime.datetime.now()
    if now > (start_time + datetime.timedelta(seconds=timeout)):
        server.should_exit = True
        raise TimeoutError("Failed to complete test within timelimit")


def test_request_aborter(open_port):
    # Get start time to ensure test doesn't hang
    start_time = datetime.datetime.now()

    # Create FastAPI app for testing
    app = FastAPI()

    # Initialize synchronization variables for tracking request process
    abort_event = threading.Event()
    request_finished = threading.Event()

    # Define an endpoint that sleeps until the client disconnects. No tests should
    # be run in this function as exceptions are caught by uvicorn
    @app.get("/test_abort")
    async def test_aborter(context: Request):
        # Create aborter and add parent event
        TEST_ABORTER = RequestAborter(context, poll_time=0.001)
        TEST_ABORTER.add_event(abort_event)

        # Wait for client to disconnect
        while not TEST_ABORTER.must_abort():
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

    # Wait for uvicorn to start
    while not server.started:
        check_server_timeout(start_time, server)
        time.sleep(0.01)

    # Try the endpoint but timeout after 1 second
    with pytest.raises(ReadTimeout):
        requests.get(
            f"http://localhost:{open_port}/test_abort",
            timeout=1,
        )

    # Wait for the request to finish/abort
    while not request_finished.is_set():
        check_server_timeout(start_time, server)
        time.sleep(0.01)

    # Assert the request aborter actually aborted
    assert test_request_aborter.TEST_ABORTER.must_abort()
    assert abort_event.is_set()
    assert request_finished.is_set()

    # Clean up the server
    server.should_exit = True
    server_thread.join()
