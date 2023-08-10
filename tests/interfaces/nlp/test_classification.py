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
from caikit.interfaces.nlp.data_model import (
    ClassificationResult,
    ClassificationResults,
    ClassificationTrainRecord,
    ClassifiedGeneratedTextResult,
    TokenClassificationResult,
    TokenClassificationResults,
)

## Setup #########################################################################

classification1 = ClassificationResult(label="temperature", score=0.71)

classification2 = ClassificationResult(label="conditions", score=0.98)

classification_result = ClassificationResults(
    results=[classification1, classification2]
)

token_classification1 = TokenClassificationResult(
    start=0, end=5, word="moose", entity="animal", score=0.8
)
token_classification2 = TokenClassificationResult(
    start=7, end=12, word="goose", entity="animal", score=0.7
)
token_classification_result = TokenClassificationResults(
    results=[token_classification1, token_classification2]
)

classification_train_record = ClassificationTrainRecord(
    text="It is 20 degrees today", labels=["temperature"]
)

classification_generated_text_result = ClassifiedGeneratedTextResult(
    text="moose goose foo bar",
    token_classification_results=[token_classification1, token_classification2],
)

## Tests ########################################################################

### ClassificationResult


def test_classification_all_fields_accessible():
    classification_result = ClassificationResult(label="temperature", score=0.71)
    assert classification_result.label == "temperature"
    assert classification_result.score == 0.71


def test_classification_from_proto_and_back():
    new = ClassificationResult.from_proto(classification1.to_proto())
    assert new.label == "temperature"
    assert new.score == 0.71


def test_classification_from_json_and_back():
    new = ClassificationResult.from_json(classification1.to_json())
    assert new.label == "temperature"
    assert new.score == 0.71


### ClassificationResults


def test_classification_result_all_fields_accessible():
    classification_result = ClassificationResults(results=[classification1])
    assert classification_result.results[0].label == "temperature"
    assert classification_result.results[0].score == 0.71


def test_classification_result_from_proto_and_back():
    new = ClassificationResults.from_proto(classification_result.to_proto())
    assert new.results[0].label == "temperature"
    assert new.results[0].score == 0.71
    assert new.results[1].label == "conditions"
    assert new.results[1].score == 0.98


def test_classification_result_from_json_and_back():
    new = ClassificationResults.from_json(classification_result.to_json())
    assert new.results[0].label == "temperature"
    assert new.results[0].score == 0.71
    assert new.results[1].label == "conditions"
    assert new.results[1].score == 0.98


### TokenClassificationResult


def test_token_classification_all_fields_accessible():
    token_classification = TokenClassificationResult(
        start=0,
        end=28,
        word="The cow jumped over the moon",
        entity="neutral",
        score=0.6,
    )
    assert token_classification.start == 0
    assert token_classification.end == 28
    assert token_classification.word == "The cow jumped over the moon"
    assert token_classification.entity == "neutral"
    assert token_classification.score == 0.6


def test_classification_from_proto_and_back():
    new = TokenClassificationResult.from_proto(token_classification1.to_proto())
    assert new.start == 0
    assert new.word == "moose"
    assert new.score == 0.8


def test_classification_from_json_and_back():
    new = TokenClassificationResult.from_json(token_classification1.to_json())
    assert new.start == 0
    assert new.word == "moose"
    assert new.score == 0.8


### TokenClassificationResults


def test_token_classification_result_all_fields_accessible():
    token_classification_result = TokenClassificationResults(
        results=[token_classification1]
    )
    assert token_classification_result.results[0].start == 0
    assert token_classification_result.results[0].word == "moose"
    assert token_classification_result.results[0].score == 0.8


def test_token_classification_result_from_proto_and_back():
    new = TokenClassificationResults.from_proto(token_classification_result.to_proto())
    assert new.results[0].start == 0
    assert new.results[0].word == "moose"
    assert new.results[0].score == 0.8
    assert new.results[1].end == 12
    assert new.results[1].entity == "animal"


def test_classification_result_from_json_and_back():
    new = TokenClassificationResults.from_json(token_classification_result.to_json())
    assert new.results[0].start == 0
    assert new.results[0].word == "moose"
    assert new.results[0].score == 0.8
    assert new.results[1].end == 12
    assert new.results[1].entity == "animal"


### ClassificationTrainRecord


def test_classification_train_record_all_fields_accessible():
    classification_train_record = ClassificationTrainRecord(
        text="It is 20 degrees today", labels=["temperature"]
    )
    assert classification_train_record.text == "It is 20 degrees today"
    assert classification_train_record.labels == ["temperature"]


def test_classification_train_record_from_proto_and_back():
    new = ClassificationTrainRecord.from_proto(classification_train_record.to_proto())
    assert new.text == "It is 20 degrees today"
    assert new.labels == ["temperature"]


def test_classification_train_record_from_json_and_back():
    new = ClassificationTrainRecord.from_json(classification_train_record.to_json())
    assert new.text == "It is 20 degrees today"
    assert new.labels == ["temperature"]


### ClassifiedGeneratedTextResult


def test_classification_generated_text_result_all_fields_accessible():
    classification_generated_text_result = ClassifiedGeneratedTextResult(
        text="moose goose foo bar",
        token_classification_results=[token_classification1, token_classification2],
    )
    assert classification_generated_text_result.text == "moose goose foo bar"
    assert (
        classification_generated_text_result.token_classification_results[0]
        == token_classification1
    )
    assert (
        classification_generated_text_result.token_classification_results[1]
        == token_classification2
    )


def test_classification_generated_text_result_from_proto_and_back():
    new = ClassifiedGeneratedTextResult.from_proto(
        classification_generated_text_result.to_proto()
    )
    assert new.text == "moose goose foo bar"
    assert new.token_classification_results[0].start == 0
    assert new.token_classification_results[0].word == "moose"
    assert new.token_classification_results[0].score == 0.8
    assert new.token_classification_results[1].start == 7
    assert new.token_classification_results[1].word == "goose"
    assert new.token_classification_results[1].score == 0.7


def test_classification_generated_text_result_from_json_and_back():
    new = ClassifiedGeneratedTextResult.from_json(
        classification_generated_text_result.to_json()
    )
    assert new.text == "moose goose foo bar"
    assert new.token_classification_results[0].start == 0
    assert new.token_classification_results[0].word == "moose"
    assert new.token_classification_results[0].score == 0.8
    assert new.token_classification_results[1].start == 7
    assert new.token_classification_results[1].word == "goose"
    assert new.token_classification_results[1].score == 0.7
