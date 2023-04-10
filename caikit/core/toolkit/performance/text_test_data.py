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

TEST_FILE_EXTENSIONS = [".txt", ".html"]


class TextDirIterator:
    """Iterator class to iterate a directory of valid files where each file's text is read
    recursively within the dir Iterator returns the raw text string on each iteration.
    """

    def __init__(self, directory):
        self.text_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(extn) for extn in TEST_FILE_EXTENSIONS):
                    self.text_files.append(os.path.join(root, file))
        self.files_iter = iter(self.text_files)

    def __iter__(self):
        return self

    def __next__(self):
        with open(
            next(self.files_iter), "r", encoding="utf8", errors="ignore"
        ) as text_file:
            return text_file.read()

    def __len__(self):
        return len(self.text_files)
