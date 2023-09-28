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

# Local
from caikit.core.data_model.streams import multipart_decoder


def test_simple_multipart_file(sample_multipart_json):
    assert multipart_decoder.is_multipart_file(sample_multipart_json)
    parts = multipart_decoder.stream_multipart_file(sample_multipart_json)

    parts = list(parts)
    assert len(parts) == 1

    byte_content = parts[0].fp.read()

    with open(sample_multipart_json, "rb") as fp:
        full_multipart_byte_content = fp.read()

    assert byte_content in full_multipart_byte_content


def test_multipart_file_with_content_header(sample_multipart_json_with_content_header):
    assert multipart_decoder.is_multipart_file(
        sample_multipart_json_with_content_header
    )
    parts = multipart_decoder.stream_multipart_file(
        sample_multipart_json_with_content_header
    )

    parts = list(parts)
    assert len(parts) == 1

    byte_content = parts[0].fp.read()

    with open(sample_multipart_json_with_content_header, "rb") as fp:
        full_multipart_byte_content = fp.read()

    assert byte_content in full_multipart_byte_content


def test_many_files_in_multipart_content(
    sample_jsonl_file, sample_csv_file, sample_json_file, tmp_path
):
    files = (sample_jsonl_file, sample_csv_file, sample_json_file)

    tmpdir = str(tmp_path)
    multipart_file = os.path.join(tmpdir, "multipart")
    boundary_string = "foobar"

    with open(multipart_file, "w") as fp:
        for file in files:
            fp.write("--")
            fp.write(boundary_string)
            fp.writelines(
                [
                    "\n",
                    f'Content-Disposition: form-data; name="{file}"; filename="{file}"\n',
                    "Content-Type: text/plain\n",
                    "\n",
                ]
            )
            with open(file, "r") as inner_fp:
                content = inner_fp.read()
            fp.write(content)
            fp.write("\n")

        fp.write("--")
        fp.write(boundary_string)
        fp.write("--")

    parts = list(multipart_decoder.stream_multipart_file(multipart_file))

    assert len(parts) == len(files)

    for actual, expected in zip(parts, files):
        actual_content = actual.fp.read()
        with open(expected, "rb") as f:
            expected_content = f.read()
        assert actual_content == expected_content
