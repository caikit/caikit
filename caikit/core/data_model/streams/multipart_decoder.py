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
See the w3 spec here: https://www.w3.org/TR/html401/interact/forms.html#h-17.13.4.2

e.g. files of the form
```
Content-Type: multipart/form-data; boundary=my_boundary

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

NB: This will also handle files which do not start with a Content-Type header.
If the file starts with a different string, we will assume that is the first boundary string.

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

# Third Party
from werkzeug import formparser
from werkzeug.datastructures import FileStorage
import werkzeug.http

# First Party
from alog import alog

# Local
from caikit.core.exceptions import error_handler

log = alog.use_channel("MULTIPART_DECODER")
error = error_handler.get(log)


@dataclasses.dataclass
class Part:
    content_type: str
    filename: str
    fp: typing.IO[bytes]


def is_multipart_file(file) -> bool:
    """Returns true if the file appears to contain a multi-part form data request"""
    log.debug3("Determining if %s is a multipart file", file)
    first_line = _get_first_nonempty_line(file)

    # Either: The beginning of the file starts with a boundary string (must start with --)
    if first_line.startswith("--"):
        log.debug3(
            "Assuming file %s is a multipart file because it begins with --", file
        )
        return True
    # Or: it's parseable as a content-type header with a boundary
    header, options = werkzeug.http.parse_options_header(first_line)
    if "multipart" not in header.lower():
        log.debug(
            "No multipart content header detected in [%s: %s], not a multipart file",
            header,
            options,
        )
        return False
    if "boundary" not in options:
        log.debug(
            "No boundary option provided in content type header [%s: %s], not a multipart file",
            header,
            options,
        )
        return False

    # Cool, should be a multipart file
    return True


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
    content. Should only be called if is_multipart_file(file) returns True"""
    first_line = _get_first_nonempty_line(file)
    if first_line.startswith("--"):
        return first_line[2:].rstrip(os.linesep)

    _, options = werkzeug.http.parse_options_header(first_line)
    return options["boundary"]


def _get_first_nonempty_line(file) -> str:
    """Return the first line of the file with content.
    Returns empty string if none exists."""
    with open(file, encoding="utf-8") as fp:
        for line in fp:
            stripped_line = line.strip()
            if stripped_line != "":
                return stripped_line
    return ""
