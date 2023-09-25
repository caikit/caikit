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


"""A decoder for files that contain multi-part form data of files.

e.g. files of the form
```
--my_boundary
Content-Disposition: form-data; name=""; filename="foo.json"
Content-Type: application/json

[
    {
        "foo": "bar"
    }
]

--my_boundary--
```

This is not meant to be an implementation detail of an HTTP handler, this is only meant to be a
general-purpose file decoder for cases where:
- A multipart form-data request has been serialized to a file
- That form data itself contained one or more files (NB: _not_ form fields)
- We wish to get a stream of those inner file contents back

Currently implemented using the `werkzeug` multipart parser, for ease of implementation.

A known drawback is that this reads the full file before streaming back the parsed file contents.
Reading multi-GB files could be slow, as this will cause serialization back to disk of the
files contained within the multipart request.
"""
# Standard
from typing import Iterator
import dataclasses
import os
import typing

# First Party
from alog import alog

# Local
from caikit.core.exceptions import error_handler

log = alog.use_channel("MULTIPART_DECODER")
error = error_handler.get(log)


# Third Party
from werkzeug import formparser
from werkzeug.datastructures import FileStorage


@dataclasses.dataclass
class Part:
    content_type: str
    filename: str
    fp: typing.IO[bytes]


def is_multipart_file(file) -> bool:
    """Returns true if the file appears to contain a multi-part form data request"""
    with open(file, "r") as fp:
        # Read a small bit of the file
        head: str = fp.read(50)

    # The beginning of the file content should start with "--"
    return head.lstrip().startswith("--")


def stream_multipart_file(file) -> Iterator[Part]:
    """Returns an iterator of Parts, where each Part comes with a content type and an io reader to
    stream the data from.

    NB: This only yields parts which are files, not other form fields.
    """

    boundary = _get_multipart_boundary(file)

    parser = formparser.MultiPartParser()
    with open(file, "rb") as fp:
        _, files = parser.parse(fp, boundary.encode(), None)

        for value in files.values():
            value: FileStorage
            # Get a readable pointer to the file
            fp = value.stream
            # Requires seeking back to the beginning because this was just written
            fp.seek(0)

            log.debug3(
                "Yielding file %s parsed from multipart file %s", value.filename, file
            )
            yield Part(content_type=value.content_type, filename=value.filename, fp=fp)


def _get_multipart_boundary(file) -> str:
    """Returns the multipart boundary string by looking for it in the first line of the file with
    content"""
    with open(file, "r") as fp:
        line = ""
        while not line:
            line = fp.readline().lstrip()
    error.value_check(
        "",
        line.startswith("--"),
        "File does not start with multipart boundary string '--'",
    )
    return line[2:].rstrip(os.linesep)
