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
"""Data structures for syntax analysis.
"""

# Standard
from enum import Enum
from typing import Dict, List, Optional
import copy
import os

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject, enums
from ....core.toolkit import fileio, isiterable
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import text_primitives

log = alog.use_channel("DATAM")


@dataobject(package="caikit_data_model.nlp")
class PartOfSpeech(Enum):
    """
    Enum for the Universal POS tag set.
    CITE: http://universaldependencies.org/u/pos/index.html
    """

    POS_UNSET = 0
    POS_ADJ = 1
    POS_ADP = 2
    POS_ADV = 3
    POS_AUX = 4
    POS_CCONJ = 5
    POS_DET = 6
    POS_INTJ = 7
    POS_NOUN = 8
    POS_NUM = 9
    POS_PART = 10
    POS_PRON = 11
    POS_PROPN = 12
    POS_PUNCT = 13
    POS_SCONJ = 14
    POS_SYM = 15
    POS_VERB = 16
    POS_X = 17


@dataobject(package="caikit_data_model.nlp")
class DependencyRelation(Enum):
    """
    Enum for the Universal Dependency Relation tag set.
    CITE: http://universaldependencies.org/u/dep/index.html

    NOTE: Subtypes were added to the data model later. Instead of commas,
    we use underscores here. This is why the numbers are in a weird order.
    """

    DEP_OTHER = 0
    DEP_ACL = 1
    DEP_ADVCL = 2
    DEP_ADVMOD = 3
    DEP_AMOD = 4
    DEP_APPOS = 5
    DEP_AUX = 6
    DEP_CASE = 7
    DEP_CC = 8
    DEP_CCOMP = 9
    DEP_CLF = 10
    DEP_COMPOUND = 11
    DEP_CONJ = 12
    DEP_COP = 13
    DEP_CSUBJ = 14
    DEP_DEP = 15
    DEP_DET = 16
    DEP_DISCOURSE = 17
    DEP_DISLOCATED = 18
    DEP_EXPL = 19
    DEP_FIXED = 20
    DEP_FLAT = 21
    DEP_GOESWITH = 22
    DEP_IOBJ = 23
    DEP_LIST = 24
    DEP_MARK = 25
    DEP_NMOD = 26
    DEP_NSUBJ = 27
    DEP_NUMMOD = 28
    DEP_OBJ = 29
    DEP_OBL = 30
    DEP_ORPHAN = 31
    DEP_PARATAXIS = 32
    DEP_PUNCT = 33
    DEP_REPARANDUM = 34
    DEP_ROOT = 35
    DEP_VOCATIVE = 36
    DEP_XCOMP = 37
    DEP_ACL_RELCL = 38
    DEP_ADVMOD_EMPH = 39
    DEP_ADVMOD_LMOD = 40
    DEP_AUX_PASS = 41
    DEP_CC_PRECONJ = 42
    DEP_CSUBJ_PASS = 43
    DEP_COMPOUND_LVC = 44
    DEP_COMPOUND_PRT = 45
    DEP_COMPOUND_REDUP = 46
    DEP_COMPOUND_SVC = 47
    DEP_DET_NUMGOV = 48
    DEP_DET_NUMNOD = 49
    DEP_DET_POSS = 50
    DEP_EXPL_IMPERS = 51
    DEP_EXPL_PASS = 52
    DEP_EXPL_PV = 53
    DEP_FLAT_FOREIGN = 54
    DEP_FLAT_NAME = 55
    DEP_NMOD_POSS = 56
    DEP_NMOD_TMOD = 57
    DEP_NSUBJ_PASS = 58
    DEP_NUMMOD_GOV = 59
    DEP_OBL_AGENT = 60
    DEP_OBL_ARG = 61
    DEP_OBL_LMOD = 62
    DEP_OBL_TMOD = 63


@dataobject(package="caikit_data_model.nlp")
class SyntaxParser(Enum):
    """
    Syntax analysis parsers that can be invoked.
    """

    SYNTAX_UNSET = 0
    SYNTAX_TOKENIZE = 1
    SYNTAX_SENTENCE = 2
    SYNTAX_LEMMA = 3
    SYNTAX_DEPENDENCY = 4
    SYNTAX_PART_OF_SPEECH = 5


error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class Dependency(DataObjectBase):
    """Structure containing information needed for a dependency parse between tokens
    within a single document."""

    relation: Annotated[DependencyRelation, FieldNumber(1)]
    identifier: Annotated[np.uint32, FieldNumber(2)]
    head: Optional[Annotated[np.uint32, FieldNumber(3)]]

    """Dependency parse annotation for a single token."""

    def __init__(self, relation, identifier, head=0):
        """Construct a new dependency parse annotation for a token.

        Args:
            relation:  int (enums.DependencyRelation)
                Enum value corresponding to a universal dependency relation.
            identifier:  int
                A unique identifier for this dependency annotation.
            head:  int
                A token identifier for the head of the dependency arc.  Zero (default) denotes the
                root of the dependency parse.
        """
        # NOTE:  We do not validate types here for performance reasons.

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(self, "_relation", relation)
        object.__setattr__(self, "_identifier", identifier)
        object.__setattr__(self, "_head", head)

    @classmethod
    def from_proto(cls, proto):
        # manual protobuf packing to optimize performance
        return cls(
            relation=DependencyRelation(proto.relation),
            identifier=proto.identifier,
            head=proto.head,
        )

    def fill_proto(self, proto):
        # manual protobuf packing to optimize performance
        proto.relation = self.relation.value
        proto.identifier = self.identifier
        proto.head = self.head
        return proto


@dataobject(package="caikit_data_model.nlp")
class Token(DataObjectBase):
    """Representation of a token and associated annotations."""

    span: Annotated[text_primitives.Span, FieldNumber(1)]
    lemma: Optional[Annotated[str, FieldNumber(3)]]
    part_of_speech: Optional[Annotated[PartOfSpeech, FieldNumber(4)]]
    dependency: Optional[Annotated[Dependency, FieldNumber(5)]]
    features: Optional[Annotated[List[str], FieldNumber(6)]]

    """A single token."""

    def __init__(
        self,
        span,
        lemma="",
        part_of_speech=PartOfSpeech.POS_UNSET,
        dependency=None,
        features=None,
    ):
        """Construct a new token.

        Args:
            span:  Span or 2-tuple (int, int)
                The location of this token in terms (begin, end) utf codepoint offsets into the
                text.
            lemma:  str
                The lemmatized form of this token.  Empty string (default) indicates that this value
                has not been set.
            part_of_speech:  (PartOfSpeech) enums.PartOfSpeech
                Enum value corresponding to a universal part-of-speech tag.  None (default)
                indicates that this value has not been set.
            dependency:  Dependency or None
                The universal dependency parse annotation for this token.  None (default) indicates
                that this value has not been set.
            features: list(str)
                Rules based features associated with this token.
        """
        # NOTE:  We do not validate types here for performance reasons.

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        # IMPORTANT - THE FUNCTIONALITY TO CREATE TOKENS FROM TUPLES WILL BE REMOVED IN V3.0;
        # THIS STATEMENT IS NOT PERFORMANT FOR LARGE TEXTS; FOR NOW WE LEAVE IT UNTIL THE NEXT
        # MAJOR RELEASE, BUT YOU SHOULD CREATE YOUR TOKENS WITH SPANS OBJECTS TO MAKE THIS SWITCH
        # EASY FOR UPDATING THE CODEBASE IN THE FUTURE VERSIONS.
        object.__setattr__(
            self,
            "_span",
            text_primitives.Span(*span) if isinstance(span, tuple) else span,
        )
        object.__setattr__(self, "_lemma", lemma)
        object.__setattr__(self, "_part_of_speech", part_of_speech)
        object.__setattr__(self, "_dependency", dependency)
        object.__setattr__(self, "_features", [] if features is None else features)

    @classmethod
    def from_proto(cls, proto):
        # manual protobuf packing to optimize performance
        span = text_primitives.Span.from_proto(proto.span)

        dependency = (
            Dependency.from_proto(proto.dependency)
            if proto.HasField("dependency")
            else None
        )

        features = list(proto.features)

        return cls(
            span=span,
            lemma=proto.lemma,
            part_of_speech=PartOfSpeech(proto.part_of_speech)
            if proto.HasField("part_of_speech")
            else None,
            dependency=dependency,
            features=features,
        )

    def fill_proto(self, proto):
        # manual protobuf packing to optimize performance
        self.span.fill_proto(proto.span)
        proto.lemma = self.lemma
        proto.part_of_speech = self.part_of_speech.value
        if self.dependency is not None:
            self.dependency.fill_proto(proto.dependency)

        if self.features is not None:
            proto.features.extend(self.features)

        return proto

    @property
    def text(self):
        """String text spanned by this token."""
        return self.span.text

    def __len__(self):
        return len(self.span)


@dataobject(package="caikit_data_model.nlp")
class Sentence(DataObjectBase):
    """Representation of a sentence and associated annotations."""

    span: Annotated[text_primitives.Span, FieldNumber(1)]

    """A single sentence."""

    def __init__(self, span):
        """Construct a new sentence.

        Args:
            span:  Span or 2-tuple (int, int)
                The location of this sentence in terms (begin, end) utf codepoint offsets into the
                text.
        """
        # NOTE:  We do not validate types here for performance reasons.

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(
            self,
            "_span",
            text_primitives.Span(*span) if isinstance(span, tuple) else span,
        )

    @classmethod
    def from_proto(cls, proto):
        # manual protobuf packing to optimize performance
        span = text_primitives.Span.from_proto(proto.span)
        return cls(span=span)

    def fill_proto(self, proto):
        # manual protobuf packing to optimize performance
        self.span.fill_proto(proto.span)
        return proto

    @property
    def text(self):
        """String text spanned by this sentence."""
        return self.span.text


@dataobject(package="caikit_data_model.nlp")
class Paragraph(DataObjectBase):
    """Representation of a paragraph and any associated annotations."""

    span: Annotated[text_primitives.Span, FieldNumber(1)]

    """A single paragraph."""

    def __init__(self, span):
        """Construct a new paragraph.

        Args:
            span:  Span or 2-tuple (int, int)
                The location of this paragraph in terms (begin, end) utf codepoint offsets into the
                text.
        """
        # NOTE:  We do not validate types here for performance reasons.

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(
            self,
            "_span",
            text_primitives.Span(*span) if isinstance(span, tuple) else span,
        )

    @classmethod
    def from_proto(cls, proto):
        # manual protobuf packing to optimize performance
        span = text_primitives.Span.from_proto(proto.span)
        return cls(span=span)

    def fill_proto(self, proto):
        # manual protobuf packing to optimize performance
        self.span.fill_proto(proto.span)
        return proto

    @property
    def text(self):
        """String text spanned by this paragraph."""
        return self.span.text


@dataobject(package="caikit_data_model.nlp")
class RawDocument(DataObjectBase):
    """Representation of a raw (unprocessed) document."""

    text: Annotated[str, FieldNumber(1)]
    producer_id: Optional[Annotated[ProducerId, FieldNumber(2)]]

    """A raw document containing some text to be processed."""

    def __init__(self, text, producer_id=None):
        """Construct a new raw document.

        Args:
            text:  str
                The string text of the document.
            producer_id:  ProducerId or None
                The block that produced this syntax analysis.
        """
        error.type_check("<NLP81353642E>", str, text=text)
        error.type_check(
            "<NLP42339847E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_producer_id", producer_id)

    @classmethod
    def load_txt(cls, filename, producer_id=None):
        """Build a raw document from a text file.

        Args:
            filename:  str
                Path to a utf8 encoded text file to load from disk.

            producer_id:  ProducerId
                The producer of this raw document.  If `None` (default) then the producer id will
                be set to the base name (not the full path) of the loaded file.
        """
        if producer_id is None:
            producer_id = ProducerId(os.path.basename(filename), version="0.0.1")

        return cls(fileio.load_txt(filename), producer_id=producer_id)

    # alias `load_txt` for backward compatibility
    from_file = load_txt


@dataobject(package="caikit_data_model.nlp")
class HTMLTagSpansList(DataObjectBase):
    span_list: Annotated[List[text_primitives.Span], FieldNumber(1)]


@dataobject(package="caikit_data_model.nlp")
class DetagPrediction(DataObjectBase):
    """HTML document representation"""

    # HTML text contained by this document
    html: Annotated[str, FieldNumber(1)]

    # Detagged text contained by this document
    text: Annotated[str, FieldNumber(2)]

    # Offsets between detagged text(index of list) to HTML text (value of index)
    # e.g. [4,5,6]:
    # text[0] -> html[4]
    # text[1] -> html[5]
    # text[2] -> html[6]
    offsets: Annotated[List[int], FieldNumber(3)]

    # Offsets where the original HTML tags were located. Specifically, they
    # point to the end of a tag (the index after ">" in "<tag>Text</tag>", so
    # `tag_offsets = [4, 14]`). When detagging, multiple HTML tags could point
    # to the same offset (for example when tags are nested "<title><p>This is
    # text</title></p>"), which is why we have removed duplicates in this list.
    tag_offsets: Optional[Annotated[List[int], FieldNumber(4)]]

    # A mapping of tag names to all the locations of such tag in the HTML
    # document, represented as a list of spans. Mainly exposed for debugging
    # purposes. For example: `{'p': [(10, 16), (20, 22)]}` would tell us that
    # the opening tag '<p>' is located at indices 10 and 20, and that the end
    # tag '</p>'s are located at positions 16 and 22, closing the corresponding
    # tags at 10 and 20, respectively.
    tag_names_to_spans: Optional[Annotated[Dict[str, HTMLTagSpansList], FieldNumber(5)]]

    # Module that produced these annotations
    producer_id: Optional[Annotated[ProducerId, FieldNumber(6)]]

    def __init__(
        self,
        html,
        text,
        offsets: List[int],
        tag_offsets: List[int] = None,
        tag_names_to_spans: Dict[str, List[text_primitives.Span]] = None,
        producer_id=None,
    ):
        """Construct a new html document.

        Args:
            html: str
                The HTML text of the document.
            text: str
                The detagged version of the HTML text of the document. Evaluated on the given HTML.
            offsets: List[int]
                List of offset mappings where
                index of list -> detagged text index
                value for index -> HTML text index
            tag_offsets: List[int]
                List of unique indices in the detagged text where the original HTML tags were
                located. Specifically, they point to the end of a tag (the index after ">" in
                "<tag>"). When detagging, multiple HTML tags could point to the same offset
                (for example when tags are nested "<title><p>This is text</title></p>"),
                which is why we have removed duplicates in this list.
            tag_names_to_spans: Dict[str, List[text_primitives.Span]]
                A mapping of tag names to all the locations of such tag in the HTML document,
                represented as a list of spans. Mainly exposed for debugging purposes.
                For example: `{'p': [(10, 16), (20, 22)]}` would tell us that the opening
                tag '<p>' is located at indices 10 and 20, and that the end tag '</p>'s are
                located at positions 16 and 22, closing the corresponding tags at 10 and 20,
                respectively.
            producer_id:  ProducerId or None
                The block that produced this object.
        """
        error.type_check("<NLP41766961E>", str, html=html, text=text)
        error.type_check_all("<NLP12001222E>", int, offsets=offsets)
        error.type_check(
            "<NLP81789848E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        error.type_check_all(
            "<NLP70514700E>", int, allow_none=True, tag_offsets=tag_offsets
        )
        error.type_check(
            "<NLP01581589E>",
            dict,
            allow_none=True,
            tag_names_to_spans=tag_names_to_spans,
        )

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(self, "_html", html)
        object.__setattr__(self, "_producer_id", producer_id)
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(
            self, "_tag_offsets", [] if tag_offsets is None else tag_offsets
        )
        object.__setattr__(
            self,
            "_tag_names_to_spans",
            {} if tag_names_to_spans is None else tag_names_to_spans,
        )

    def remap(self, detagged_doc):
        """Return a remapped object whose watson_nlp.data_model.Span objects are updated if found in
        input_doc (obtained from detagged text) taking its corresponding span offsets from its HTML
        text.

        Args:
            detagged_doc: watson_nlp.data_model.base.DataBase
                Source document object, its spans are unchanged.

        Returns:
            watson_nlp.data_model.base.DataBase (same object type as input_doc)
                Deep copy of input_doc with spans updated to match with HTML offsets.

        Notes:
            HTML: <html>Text</html> has Span (6,10)
            Its detagged representation: Text has the span (0,4)

            For a given input_doc which is any object specified in watson_nlp.data_model which has a
            `Span`, the plain text span offsets are replaced with the html span offset and the
            `text` attribute is replaced with the html text in the `remapped_doc`.
        """
        if self.offsets:
            remapped_doc = copy.deepcopy(detagged_doc)
            self._update_spans(remapped_doc, detagged_doc)
            if hasattr(detagged_doc, "text"):
                remapped_doc.text = self.html
            return remapped_doc
        return detagged_doc

    def _update_spans(self, remap_obj, detagged_obj):
        """For every iterable object except str, do a depth first search on Span simultaneously on
        the source document and the deep copy, replacing the deep copy spans with the source
        document spans.

        We iterate on both since different iterated objects may be re-used in nested structure, and
        their original spans may be lost when done sequentially.

        Args:
            remap_obj: watson_nlp.data_model.base.DataBase
                Deep copy of the input document object, where spans are to be updated.
            detagged_obj: watson_nlp.data_model.base.DataBase
                Source document object, its spans are unchanged.
        """
        if detagged_obj and isinstance(detagged_obj, text_primitives.Span):
            try:
                # Detagged offsets have a 1-1 mapping with *inclusive* spans
                # e.g. for offsets [2,3,6], original map of [0,2) could end up with [2,6)
                # yielded unwanted span extending to [4,5],
                # so we fix by looking for [offset[0], offset[2 - 1] + 1): [2,4)
                remap_obj.begin = self.offsets[detagged_obj.begin]
                if detagged_obj.begin == detagged_obj.end:  # Empty Spans
                    remap_obj.end = remap_obj.begin
                else:
                    remap_obj.end = self.offsets[detagged_obj.end - 1] + 1
                remap_obj.text = self.html[remap_obj.begin : remap_obj.end]
            except IndexError:
                # Invalid input doc, no matching offset to remap
                error(
                    "<NLP13937860E>",
                    ValueError(
                        "Invalid input document. Matching spans not found in HTML"
                    ),
                )
        else:
            if not isinstance(detagged_obj, str) and isiterable(detagged_obj):
                for remap_item, detagged_item in zip(remap_obj, detagged_obj):
                    self._update_spans(remap_item, detagged_item)
            else:
                # For every 'field' object present in the data_model (specified in the .proto files)
                if hasattr(detagged_obj, "fields"):
                    for field in detagged_obj.fields:
                        self._update_spans(
                            getattr(remap_obj, field), getattr(detagged_obj, field)
                        )


@dataobject(package="caikit_data_model.nlp")
class SyntaxParserSpec(DataObjectBase):
    parsers: Annotated[List[SyntaxParser], FieldNumber(1)]

    """A parser specification informs a syntax analysis which syntax annotations should be
    constructed.
    """

    # map from izumo parser names to our parser enums
    izumo_map = {
        "token": enums.SyntaxParser["SYNTAX_TOKENIZE"],
        "sentence": enums.SyntaxParser["SYNTAX_SENTENCE"],
        "lemma": enums.SyntaxParser["SYNTAX_LEMMA"],
        "part_of_speech": enums.SyntaxParser["SYNTAX_PART_OF_SPEECH"],
        "dependency": enums.SyntaxParser["SYNTAX_DEPENDENCY"],
    }
    # reverse map back to izumo names
    izumo_map_rev = {value: key for key, value in izumo_map.items()}

    def __init__(self, parsers):
        """Construct a new syntax parser specification.

        Args:
            parsers:  list(str) or list(enums.SyntaxParser)
                A list of parsers used to construct syntax annotations.  Can be any of the values in
                the enums.SyntaxParser enum or a list containing any of the following strings:
                'token', 'sentence', 'lemma', 'part_of_speech', 'dependency'
        """
        error.type_check_all("<NLP01572687E>", str, int, parsers=parsers)

        self.parsers = []
        for parser in parsers:
            # if parser is an integer, assume its an enum
            if isinstance(parser, int):
                self.parsers.append(parser)
                continue

            # check if its the string form of the enum
            parser_num = enums.SyntaxParser.get(parser)
            if parser_num == 0:
                error(
                    "<NLP71783982E>", ValueError("parser `{}` is unset".format(parser))
                )

            # check if its the izumo string parser name
            if parser_num is None:
                parser_num = self.izumo_map.get(parser.lower())

            # if still not found, raise an exception
            if parser_num is None:
                error(
                    "<NLP71783995E>",
                    ValueError("invalid parser `{}` specified".format(parser)),
                )

            self.parsers.append(parser_num)

    def to_izumo(self):
        """Convert to an Izumo compatible parser specification."""
        return [self.izumo_map_rev[parser] for parser in self.parsers]


@dataobject(package="caikit_data_model.nlp")
class SyntaxPrediction(DataObjectBase):
    """Representation of a document that has had syntax analysis applied
    potentially including tokenization, sentence splitting NP chunking,
    dependency parsing, paragraph splitting, et cetra."""

    text: Annotated[str, FieldNumber(1)]
    producer_id: Optional[Annotated[ProducerId, FieldNumber(2)]]
    tokens: Optional[Annotated[List[Token], FieldNumber(3)]]
    sentences: Optional[Annotated[List[Sentence], FieldNumber(4)]]
    paragraphs: Optional[Annotated[List[Paragraph], FieldNumber(6)]]

    """A syntax analysis container.  This class contains all elements of a syntax analysis,
    including tokenization, sentences, dependency parsing, parts-of-speech, and more.
    """

    def __init__(
        self,
        text,
        producer_id=None,
        tokens=None,
        sentences=None,
        paragraphs=None,
        slice_and_set_span_texts=True,
    ):
        """Construct a new syntax analysis container from it's components.

        Args:
            text:  str or RawDocument
                The text analyzed in this syntax analysis.
            producer_id:  ProducerId or None
                The block that produced this syntax analysis.
            tokens:  list(Token) or None
                The tokens contained in this syntax analysis.
                Default to empty list.
            sentences:  list(Sentence)
                The sentences contained in this syntax analysis.
                Default to empty list.
            paragraphs:  list(Paragraphs)
                The paragraphs contained in this syntax analysis.
                Default to empty list.
            slice_and_set_span_texts: bool
                Set span texts to the given tokens, sentences, and paragraphs
        """
        error.type_check("<NLP23782538E>", str, RawDocument, text=text)
        error.type_check(
            "<NLP39283031E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        error.type_check_all("<NLP10113512E>", Token, allow_none=True, tokens=tokens)
        error.type_check_all(
            "<NLP59052768E>", Sentence, allow_none=True, sentences=sentences
        )
        error.type_check_all(
            "<NLP82056954E>", Paragraph, allow_none=True, paragraphs=paragraphs
        )

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(
            self, "_text", text.text if isinstance(text, RawDocument) else text
        )
        object.__setattr__(self, "_producer_id", producer_id)

        if slice_and_set_span_texts:
            object.__setattr__(
                self, "_tokens", self._slice_and_set_span_texts(tokens, text)
            )
            object.__setattr__(
                self, "_sentences", self._slice_and_set_span_texts(sentences, text)
            )
            object.__setattr__(
                self, "_paragraphs", self._slice_and_set_span_texts(paragraphs, text)
            )
        else:
            object.__setattr__(self, "_tokens", [] if tokens is None else tokens)
            object.__setattr__(
                self, "_sentences", [] if sentences is None else sentences
            )
            object.__setattr__(
                self, "_paragraphs", [] if paragraphs is None else paragraphs
            )

    @staticmethod
    def _slice_and_set_span_texts(items, text):
        """Sort a list of data structures with .span attributes of type Span and extract the
        spanned text.  This is a helper function for internal use.

        Args:
            items:  list(data_model.object)
                A list of data model objects that have a Span attribute .span.
            text:  str
                The full document text that our spans are in reference to.
                The Span.slice_and_set_text method will be called to fill the span in .span.

        Returns:
            list(data_model.object)
                A list containing the items in sorted order and with extracted text.
        """
        if not items:
            return []

        # automatically extract text for each span
        text_length = len(text)
        for item in items:
            item.span.slice_and_set_text(text, text_length)

        return items

    def get_token_texts(self):
        """Get a list(str) containing the texts for all tokens."""
        return [token.span.text for token in self.tokens]

    def get_token_spans(self):
        """Get a list((begin, end)) utf codepoint spans for all tokens."""
        return [token.span() for token in self.tokens]

    @alog.logged_function(log.debug2)
    def get_tokens_by_sentence(self, max_sent_toks=None):
        """Get tokens nested by sentence.

        Args:
            max_sent_toks:  None or int
                Maximum number of tokens allowed in any single sentence.  Sentences that contain
                more than max_sent_toks tokens will be split.  If None (default) then there is no
                maximum.  NOTE:  setting max_sent_toks may result in more sentences being returned
                than are present in the overall SyntaxPrediction because sentences can be split.
                This functionality is useful for algorithms that rely on sentence-level batching
                and may encounter performance issues when processing very long sentences.

        Returns:
            list(list(dm.Token))
                A list of sentences each composed of a list of tokens.
        """
        error.value_check(
            "<NLP33121328E>",
            max_sent_toks is None or max_sent_toks > 0,
            "`max_sent_toks` of `{}` should be `None` or integer greater than zero",
            max_sent_toks,
        )

        token_iter = iter(self.tokens)
        sentence_iter = iter(self.sentences)
        sentences = [[]]

        cur_token = next(token_iter, None)
        cur_sentence = next(sentence_iter, None)

        try:
            # Default None if sentence is empty in case of empty doc
            if cur_sentence is not None:
                # we assume tokens and sentences are sorted by begin offset
                while cur_token is not None:
                    # check if token is contained in current sentence
                    # if so, add to sentence list and get next token
                    if cur_token.span in cur_sentence.span:
                        # if maximum sentence length is exceeded, split off new sentence
                        if max_sent_toks and len(sentences[-1]) >= max_sent_toks:
                            sentences.append([])

                        sentences[-1].append(cur_token)
                        cur_token = next(token_iter, None)

                    # otherwise, move on to next sentence
                    else:
                        sentences.append([])
                        # no default since we need to failover when sentence boundaries are exceeded
                        cur_sentence = next(sentence_iter)

        # raise error if sentences exhausted before tokens
        except StopIteration:
            error(
                "<NLP71784010E>",
                ValueError(
                    "token `{}` not contained in any sentences".format(cur_token.span())
                ),
            )

        return sentences

    @alog.logged_function(log.debug2)
    def get_tokens_by_span(self, span):
        """Gets the tokens that are in the given span.

        Args:
            span:  text_primitives.Span or 2-tuple
                A span of text to examine for tokens.

        Returns:
            list(data_model.Token)
                A list of tokens that are contained within the specified span.

        FIXME:
            The current implementation only works if the span argument begins on a token boundary.
            Otherwise, the empty list will be returned.
        """
        if isinstance(span, tuple):
            span = text_primitives.Span(*span)

        # FIXME: should return tokens contained in span even if the first token is not exact.
        #   For example, if span=(10, 20) and tokens=[(12, 15), (15, 18)]
        tokens = []
        index = self.find_token(span.begin)
        if index == -1:
            return tokens

        while index < len(self.tokens) and self.tokens[index].span.end <= span.end:
            cur_token = self.tokens[index]
            tokens.append(cur_token)
            index = index + 1

        return tokens

    @alog.logged_function(log.debug2)
    def find_token(self, begin_offset):
        """Find the index of a token with a given begin offset using binary search.

        Args:
            begin_offset: int
                The begin offset of the token to search for.

        Returns:
            int
                The integer index into the token list containing a token with the specified begin
                offset.  If the token is not found, -1 is returned.

        Notes:
            This search only checks for exact matches for begin offsets.
        """
        min_index = 0
        max_index = len(self.tokens) - 1

        while min_index <= max_index:
            cur_index = int((min_index + max_index) / 2)
            cur_element = self.tokens[cur_index]

            if cur_element.span.begin < begin_offset:
                min_index = cur_index + 1
            elif cur_element.span.begin > begin_offset:
                max_index = cur_index - 1
            elif cur_element.span.begin == begin_offset:
                return cur_index

        return -1

    def get_token_texts_by_sentence(self, max_sent_toks=None):
        """Get token texts nested by sentence.

        Args:
            max_sent_toks:  None or int
                Maximum number of tokens allowed in any single sentence.  Sentences that contain
                more than max_sent_toks tokens will be split.  If None (default) then there is no
                maximum.  NOTE:  setting max_sent_toks may result in more sentences being returned
                than are present in the overall SyntaxPrediction because sentences can be split.
                This functionality is useful for algorithms that rely on sentence-level batching
                and may encounter performance issues when processing very long sentences.

        Returns:
            list(list(str))
                A list of sentences each composed of a list of token text strings.
        """
        return [
            [token.span.text for token in tokens]
            for tokens in self.get_tokens_by_sentence(max_sent_toks=max_sent_toks)
        ]

    def get_token_spans_by_sentence(self, max_sent_toks=None):
        """Get token span tuples nested by sentence.

        Args:
            max_sent_toks:  None or int
                Maximum number of tokens allowed in any single sentence.  Sentences that contain
                more than max_sent_toks tokens will be split.  If None (default) then there is no
                maximum.  NOTE:  setting max_sent_toks may result in more sentences being returned
                than are present in the overall SyntaxPrediction because sentences can be split.
                This functionality is useful for algorithms that rely on sentence-level batching
                and may encounter performance issues when processing very long sentences.

        Returns:
            list(list((int, int)))
                A list of sentences each composed of a list of token spans represented as
                (begin, end) 2-tuples.
        """
        return [
            [token.span() for token in tokens]
            for tokens in self.get_tokens_by_sentence(max_sent_toks=max_sent_toks)
        ]

    def get_sentence_texts(self):
        """Get a list(str) containing the texts for all sentences."""
        return [sentence.span.text for sentence in self.sentences]

    def get_POS_texts(self):
        """Get a list(str) containing the part of speech for all tokens."""
        return [
            enums.PartOfSpeechRev[token.part_of_speech.value] for token in self.tokens
        ]

    def get_sentence_spans(self):
        """Get a list((begin, end)) utf codepoint spans for all sentences."""
        return [sentence.span() for sentence in self.sentences]

    def get_sentence_span_primitives(self):
        """Get a list(primitives.Span object) for all sentences."""
        return [sentence.span for sentence in self.sentences]

    def get_paragraph_texts(self):
        """Get a list(str) containing the texts for all paragraphs."""
        return [paragraph.span.text for paragraph in self.paragraphs]

    def get_paragraph_spans(self):
        """Get a list((begin, end)) utf codepoint spans for all paragraphs."""
        return [paragraph.span() for paragraph in self.paragraphs]

    def get_sentence_containing_span(self, span):
        """Get sentence which completely contains a given span (sentence.begin <= span.begin and
        sentence.end >= span.end)

        Args:
            span: data_model.Span
                Span contained within a sentence

        Returns:
            data_model.Sentence
                Sentence which contains the span, None if no sentence is found
            boolean
                False if the sentence returned fully contains the span,
                True if the given span overlaps in multiple sentences
        """
        min_index = 0
        max_index = len(self.sentences) - 1
        overlap_found = False

        while min_index <= max_index:
            mid_index = int(min_index + (max_index - min_index) / 2)
            sentence = self.sentences[mid_index]
            if span in sentence.span:
                return sentence, overlap_found
            if span.overlaps(sentence.span):
                overlap_found = True
                return sentence, overlap_found
            if span > sentence.span:
                min_index += 1
            else:
                max_index -= 1
        return None, overlap_found

    def __len__(self):
        """The length, in utf codepoints, of the text in this SyntaxPrediction."""
        return len(self.text)
