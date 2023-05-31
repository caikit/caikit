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
"""Common data model containing all data structures that are passed in and out of blocks.
"""
# Local
from . import (
    abstractive_summary,
    categories,
    classification,
    clustering,
    concepts,
    embedding,
    emotion,
    entities,
    keywords,
    lang_detect,
    matrix,
    noun_phrases,
    relations,
    rule_based_response,
    sentiment,
    syntax,
    target_mentions,
    text_generation,
    text_primitives,
    text_similarity,
    topics,
    vectorization,
)
from .abstractive_summary import AbstractiveSummary
from .categories import CategoriesPrediction, Category, RelevantText
from .classification import (
    ClassificationPrediction,
    ClassificationTrainRecord,
    ClassInfo,
    ModelType,
)
from .clustering import ClusteringPrediction
from .concepts import Concept, ConceptsPrediction
from .embedding import Embedding, EmbeddingPrediction
from .emotion import (
    AggregatedEmotionPrediction,
    Emotion,
    EmotionMention,
    EmotionPrediction,
)
from .entities import (
    EntitiesPrediction,
    Entity,
    EntityDisambiguation,
    EntityMention,
    EntityMentionAnnotation,
    EntityMentionClass,
    EntityMentionsPrediction,
    EntityMentionsTrainRecord,
    EntityMentionType,
)
from .keywords import Keyword, KeywordsPrediction
from .lang_detect import LangCode, LangDetectPrediction
from .matrix import DenseMatrix, SparseMatrix
from .noun_phrases import NounPhrase, NounPhrasesPrediction
from .relations import (
    Relation,
    RelationMention,
    RelationMentionsPrediction,
    RelationsPrediction,
)
from .rule_based_response import (
    PropertyListValueBool,
    PropertyListValueFloat,
    PropertyListValueInt,
    PropertyListValueSpan,
    PropertyListValueStr,
    RulesPrediction,
    View,
    ViewProperty,
    ViewPropertyValue,
)
from .sentiment import (
    AggregatedSentiment,
    AggregateSentimentPrediction,
    AtomicSentiment,
    Sentiment,
    SentimentLabel,
    SentimentMention,
    SentimentPrediction,
    SentimentProb,
    TargetsSentimentPrediction,
)
from .syntax import (
    Dependency,
    DependencyRelation,
    DetagPrediction,
    HTMLTagSpansList,
    Paragraph,
    PartOfSpeech,
    RawDocument,
    Sentence,
    SyntaxParser,
    SyntaxParserSpec,
    SyntaxPrediction,
    Token,
)
from .target_mentions import TargetMentions, TargetMentionsPrediction, TargetPhrases
from .text_generation import GeneratedResult, StopReason
from .text_primitives import NGram, Span
from .text_similarity import TextSimilarityPrediction
from .topics import Topic, TopicPhrase, TopicsPrediction
from .vectorization import Vectorization, VectorizationPrediction
