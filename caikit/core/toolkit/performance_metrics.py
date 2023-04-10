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

"""Helper functions for performance computations.
"""


def kilo_chars_per_second(text_len, iterations, seconds):
    return text_len * iterations / 1000 / seconds


def kilo_chars_per_second_text(text, iterations, seconds):
    return kilo_chars_per_second(len(text), iterations, seconds)


def iterations_per_second(iterations, seconds):
    return iterations / seconds
