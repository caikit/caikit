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
from typing import List

# Local
from caikit.core import TaskBase, task
from caikit.interfaces.nlp.data_model import SentenceListScores, SentenceScores


@task(
    required_parameters={"source_sentence": str, "sentences": List[str]},
    output_type=SentenceScores,
)
class SentenceSimilarityTask(TaskBase):
    """Compare the source_sentence to each of the sentences.
    Result contains a list of scores in the order of the input sentences.
    """


@task(
    required_parameters={"source_sentences": List[str], "sentences": List[str]},
    output_type=SentenceListScores,
)
class SentenceSimilarityTasks(TaskBase):
    """Compare each of the source_sentences to each of the sentences.
    Returns a list of results in the order of the source_sentences.
    Each result contains a list of scores in the order of the input sentences.
    """
