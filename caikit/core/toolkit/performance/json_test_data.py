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
import json


class JsonDataIterator:
    """Iterator class to iterate a ".json" file where each line is a json element with "label" and
    "text" attribute which is a format chosen to comply with:
    https://github.com/caikit-nlu/nlp-core-resources/blob/master/np-chunker/src/test/resources
    /docs/common/en/performance/en-50k-200.json
    Iterator returns the raw text string on each iteration
    """

    def __init__(self, directory):
        with open(directory, encoding="utf8", errors="ignore") as test_json_spec_file:
            self.files = test_json_spec_file.readlines()
            self.files_iter = iter(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        file = json.loads(next(self.files_iter))
        if "text" in file:
            return file["text"]

    def __len__(self):
        return len(self.files)
