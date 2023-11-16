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
from . import (
    classification,
    embedding_vectors,
    package,
    reranker,
    sentence_similarity,
    text,
    text_generation,
)
from .classification import (
    ClassificationResult,
    ClassificationResults,
    ClassificationTrainRecord,
    ClassifiedGeneratedTextResult,
    ClassifiedGeneratedTextStreamResult,
    TokenClassificationResult,
    TokenClassificationResults,
    TokenClassificationStreamResult,
)
from .embedding_vectors import EmbeddingResult, EmbeddingResults
from .package import NLP_PACKAGE
from .reranker import RerankResult, RerankResults, RerankScore, RerankScores
from .sentence_similarity import (
    SentenceSimilarityResult,
    SentenceSimilarityResults,
    SentenceSimilarityScores,
)
from .text import Token, TokenizationResults, TokenizationStreamResult
from .text_generation import (
    FinishReason,
    GeneratedTextResult,
    GeneratedTextStreamResult,
    GeneratedToken,
    TokenStreamDetails,
)
