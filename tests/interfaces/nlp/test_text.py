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
from caikit.interfaces.nlp.data_model import Token, TokenizationResults

## Setup #########################################################################

dummy_token = Token(start=0, end=11, text="Hello World")
dummy_token2 = Token(start=0, end=7, text="foo bar")
tokenization_result = TokenizationResults([dummy_token, dummy_token2])

## Tests ########################################################################

### Token


def test_token_all_fields_accessible():
    token = Token(start=0, end=11, text="Hello World")
    assert token.start == 0
    assert token.end == 11
    assert token.text == "Hello World"


def test_token_from_proto_and_back():
    new = Token.from_proto(dummy_token.to_proto())
    assert new.start == 0
    assert new.end == 11
    assert new.text == "Hello World"


def test_token_from_json_and_back():
    new = Token.from_json(dummy_token.to_json())
    assert new.start == 0
    assert new.end == 11
    assert new.text == "Hello World"


### TokenizationResults


def test_tokenization_result_all_fields_accessible():
    tokenization_result = TokenizationResults([dummy_token, dummy_token2])
    assert tokenization_result.results[0] == dummy_token
    assert tokenization_result.results[1] == dummy_token2


def test_tokenization_result_from_proto_and_back():
    new = TokenizationResults.from_proto(tokenization_result.to_proto())
    assert new.results[0] == dummy_token
    assert new.results[1] == dummy_token2


def test_tokenization_result_from_json_and_back():
    new = TokenizationResults.from_json(tokenization_result.to_json())
    assert new.results[0] == dummy_token
    assert new.results[1] == dummy_token2
