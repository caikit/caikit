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

# Local
from caikit.interfaces.nlp.data_model import FinishReason, GeneratedTextResult

## Setup #########################################################################

dummy_generated_response = GeneratedTextResult(
    generated_text="foo bar", generated_tokens=2, finish_reason=FinishReason.TIME_LIMIT
)

## Tests ########################################################################

### Generated Text Result
def test_all_fields_accessible():
    generated_response = GeneratedTextResult(
        generated_text="foo bar",
        generated_tokens=2,
        finish_reason=FinishReason.STOP_SEQUENCE,
    )
    # assert all((hasattr(obj, field) for field in obj.fields))
    assert generated_response.generated_text == "foo bar"
    assert generated_response.generated_tokens == 2
    assert generated_response.finish_reason == FinishReason.STOP_SEQUENCE


def test_from_proto_and_back():
    new = GeneratedTextResult.from_proto(dummy_generated_response.to_proto())
    assert new.generated_text == "foo bar"
    assert new.generated_tokens == 2
    assert new.finish_reason == FinishReason.TIME_LIMIT.value


def test_from_json_and_back():
    new = GeneratedTextResult.from_json(dummy_generated_response.to_json())
    assert new.generated_text == "foo bar"
    assert new.generated_tokens == 2
    assert new.finish_reason == FinishReason.TIME_LIMIT.value
