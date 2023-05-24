# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Standard
import os

# Local
from . import utils
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit


class TestDependency(TestCaseBase):
    def setUp(self):
        self.dependency = dm.Dependency(
            relation=dm.enums.DependencyRelation.DEP_OBJ, identifier=0, head=99
        )

        self.dependency_minimal = dm.Dependency(
            relation=dm.enums.DependencyRelation["DEP_ACL"], identifier=1
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.dependency))
        self.assertTrue(utils.validate_fields(self.dependency_minimal))

    def test_from_proto_and_back(self):
        new = dm.Dependency.from_proto(self.dependency.to_proto())
        self.assertEqual(new.relation, self.dependency.relation)
        self.assertEqual(new.identifier, self.dependency.identifier)
        self.assertEqual(new.head, self.dependency.head)

        new = dm.Dependency.from_proto(self.dependency_minimal.to_proto())
        self.assertEqual(new.relation, self.dependency_minimal.relation)
        self.assertEqual(new.identifier, self.dependency_minimal.identifier)
        self.assertEqual(new.head, self.dependency_minimal.head)

    def test_from_json_and_back(self):
        new = dm.Dependency.from_json(self.dependency.to_json())
        self.assertEqual(new.relation, self.dependency.relation)
        self.assertEqual(new.identifier, self.dependency.identifier)
        self.assertEqual(new.head, self.dependency.head)

        new = dm.Dependency.from_json(self.dependency_minimal.to_json())
        self.assertEqual(new.relation, self.dependency_minimal.relation)
        self.assertEqual(new.identifier, self.dependency_minimal.identifier)
        self.assertEqual(new.head, self.dependency_minimal.head)


class TestToken(TestCaseBase):
    def setUp(self):
        dependency = dm.Dependency(
            relation=dm.enums.DependencyRelation.DEP_ACL, identifier=1
        )
        self.token = dm.Token(
            dm.Span(0, 7, text="testing"),
            lemma="test",
            part_of_speech=dm.enums.PartOfSpeech.POS_ADJ,
            dependency=dependency,
        )

        self.token_minimal = dm.Token(dm.Span(0, 10))

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.token))
        self.assertTrue(utils.validate_fields(self.token_minimal))

    def test_from_proto_and_back(self):
        new = dm.Token.from_proto(self.token.to_proto())
        self.assertEqual(new.span.begin, self.token.span.begin)
        self.assertEqual(new.span.end, self.token.span.end)
        self.assertEqual(new.lemma, self.token.lemma)
        self.assertEqual(new.lemma, self.token.lemma)
        self.assertEqual(new.part_of_speech, self.token.part_of_speech)
        self.assertEqual(new.part_of_speech, self.token.part_of_speech)
        self.assertNotEqual(new.dependency, None)
        self.assertNotEqual(self.token.dependency, None)

        new = dm.Token.from_proto(self.token_minimal.to_proto())
        self.assertEqual(new.span.begin, self.token_minimal.span.begin)
        self.assertEqual(new.span.end, self.token_minimal.span.end)
        self.assertEqual(new.lemma, self.token_minimal.lemma)
        self.assertEqual(new.lemma, self.token_minimal.lemma)
        self.assertEqual(new.part_of_speech, self.token_minimal.part_of_speech)
        self.assertEqual(new.part_of_speech, self.token_minimal.part_of_speech)
        self.assertEqual(new.dependency, None)
        self.assertEqual(self.token_minimal.dependency, None)

    def test_from_json_and_back(self):
        new = dm.Token.from_json(self.token.to_json())
        self.assertEqual(new.span.begin, self.token.span.begin)
        self.assertEqual(new.span.end, self.token.span.end)
        self.assertEqual(new.lemma, self.token.lemma)
        self.assertEqual(new.lemma, self.token.lemma)
        self.assertEqual(new.part_of_speech, self.token.part_of_speech)
        self.assertEqual(new.part_of_speech, self.token.part_of_speech)
        self.assertNotEqual(new.dependency, None)
        self.assertNotEqual(self.token.dependency, None)

        new = dm.Token.from_json(self.token_minimal.to_json())
        self.assertEqual(new.span.begin, self.token_minimal.span.begin)
        self.assertEqual(new.span.end, self.token_minimal.span.end)
        self.assertEqual(new.lemma, self.token_minimal.lemma)
        self.assertEqual(new.part_of_speech, self.token_minimal.part_of_speech)
        self.assertEqual(new.dependency, None)
        self.assertEqual(self.token_minimal.dependency, None)


class TestSentence(TestCaseBase):
    def setUp(self):
        self.sentence = dm.Sentence(dm.Span(0, 11, text="Hello World"))
        self.sentence_minimal = dm.Sentence((0, 20))

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.sentence))
        self.assertTrue(utils.validate_fields(self.sentence_minimal))

    def test_from_proto_and_back(self):
        new = dm.Sentence.from_proto(self.sentence.to_proto())
        self.assertEqual(new.span.begin, self.sentence.span.begin)
        self.assertEqual(new.span.end, self.sentence.span.end)

        new = dm.Sentence.from_proto(self.sentence_minimal.to_proto())
        self.assertEqual(new.span.begin, self.sentence_minimal.span.begin)
        self.assertEqual(new.span.end, self.sentence_minimal.span.end)

    def test_from_json_and_back(self):
        new = dm.Sentence.from_json(self.sentence.to_json())
        self.assertEqual(new.span.begin, self.sentence.span.begin)
        self.assertEqual(new.span.end, self.sentence.span.end)

        new = dm.Sentence.from_json(self.sentence_minimal.to_json())
        self.assertEqual(new.span.begin, self.sentence_minimal.span.begin)
        self.assertEqual(new.span.end, self.sentence_minimal.span.end)


class TestParagraph(TestCaseBase):
    def setUp(self):
        self.paragraph = dm.Paragraph(dm.Span(0, 11, text="Hello World"))
        self.paragraph_minimal = dm.Paragraph((0, 20))

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.paragraph))
        self.assertTrue(utils.validate_fields(self.paragraph_minimal))

    def test_from_proto_and_back(self):
        new = dm.Paragraph.from_proto(self.paragraph.to_proto())
        self.assertEqual(new.span.begin, self.paragraph.span.begin)
        self.assertEqual(new.span.end, self.paragraph.span.end)

        new = dm.Paragraph.from_proto(self.paragraph_minimal.to_proto())
        self.assertEqual(new.span.begin, self.paragraph_minimal.span.begin)
        self.assertEqual(new.span.end, self.paragraph_minimal.span.end)

    def test_from_json_and_back(self):
        new = dm.Paragraph.from_json(self.paragraph.to_json())
        self.assertEqual(new.span.begin, self.paragraph.span.begin)
        self.assertEqual(new.span.end, self.paragraph.span.end)

        new = dm.Paragraph.from_json(self.paragraph_minimal.to_json())
        self.assertEqual(new.span.begin, self.paragraph_minimal.span.begin)
        self.assertEqual(new.span.end, self.paragraph_minimal.span.end)


class TestRawDocument(TestCaseBase):
    def setUp(self):
        self.raw_doc = dm.RawDocument(
            "Hello World!", producer_id=dm.ProducerId("Test", "1.0")
        )
        self.raw_doc_minimal = dm.RawDocument("Hello World!")
        self.linux_fixture_path = os.path.join(self.fixtures_dir, "linux.txt")
        self.raw_doc_long = dm.RawDocument.load_txt(self.linux_fixture_path)

    def test_load_txt(self):
        load_txt_doc = dm.RawDocument.load_txt(self.linux_fixture_path)
        from_file_doc = dm.RawDocument.from_file(self.linux_fixture_path)

        self.assertEqual(load_txt_doc.text, from_file_doc.text)
        self.assertEqual(load_txt_doc.producer_id.name, "linux.txt")

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.raw_doc))
        self.assertTrue(utils.validate_fields(self.raw_doc_minimal))
        self.assertTrue(utils.validate_fields(self.raw_doc_long))

    def test_from_proto_and_back(self):
        new = dm.RawDocument.from_proto(self.raw_doc.to_proto())
        self.assertEqual(new.text, self.raw_doc.text)
        self.assertEqual(new.producer_id.name, self.raw_doc.producer_id.name)
        self.assertEqual(new.producer_id.version, self.raw_doc.producer_id.version)

        new = dm.RawDocument.from_proto(self.raw_doc_minimal.to_proto())
        self.assertEqual(new.text, self.raw_doc_minimal.text)
        self.assertEqual(new.producer_id, None)

    def test_from_json_and_back(self):
        new = dm.RawDocument.from_json(self.raw_doc.to_json())
        self.assertEqual(new.text, self.raw_doc.text)
        self.assertEqual(new.producer_id.name, self.raw_doc.producer_id.name)
        self.assertEqual(new.producer_id.version, self.raw_doc.producer_id.version)

        new = dm.RawDocument.from_json(self.raw_doc_minimal.to_json())
        self.assertEqual(new.text, self.raw_doc_minimal.text)
        self.assertEqual(new.producer_id, None)


class TestDetagPrediction(TestCaseBase):
    def setUp(self):
        html_raw = "<html><body><p>Sample <b>Text</b></p></body></html>"
        self.html_doc = dm.DetagPrediction(
            html=html_raw,
            text="Sample Text",
            offsets=[i + html_raw.find("Sample ") for i in range(len("Sample "))]
            + [i + html_raw.find("Text") for i in range(len("Text"))],
            tag_offsets=[i for (i, ch) in enumerate(html_raw) if ch == ">"],
        )
        self.text_doc = self.html_doc.text
        self.syntax = dm.SyntaxPrediction(
            text=self.html_doc.text,
            producer_id=dm.ProducerId("Test", "1.0.0"),
            tokens=[
                dm.Token(dm.Span(0, 6, "Sample")),
                dm.Token(dm.Span(7, 11, "Text")),
            ],
            sentences=[dm.Sentence((0, 11, "Sample Text"))],
            paragraphs=[dm.Paragraph((0, 11, "Sample Text"))],
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.html_doc))

    def test_from_proto_and_back(self):
        html_doc_clone = dm.DetagPrediction.from_proto(self.html_doc.to_proto())
        self.assertEqual(html_doc_clone.html, self.html_doc.html)
        self.assertEqual(html_doc_clone.text, self.html_doc.text)
        self.assertEqual(html_doc_clone.offsets, self.html_doc.offsets)

    def test_from_from_and_back(self):
        html_doc_clone = dm.DetagPrediction.from_json(self.html_doc.to_json())
        self.assertEqual(html_doc_clone.html, self.html_doc.html)
        self.assertEqual(html_doc_clone.text, self.html_doc.text)
        self.assertEqual(html_doc_clone.offsets, self.html_doc.offsets)

    def test_remap(self):
        remapped_syntax = self.html_doc.remap(self.syntax)
        self.assertEqual(
            "<html><body><p>Sample <b>Text</b></p></body></html>", self.html_doc.html
        )
        self.assertEqual("Sample Text", self.html_doc.text)
        self.assertEqual(
            (0, 6), (self.syntax.tokens[0].span.begin, self.syntax.tokens[0].span.end)
        )
        self.assertEqual(
            (7, 11), (self.syntax.tokens[1].span.begin, self.syntax.tokens[1].span.end)
        )
        self.assertEqual(
            (0, 11),
            (self.syntax.sentences[0].span.begin, self.syntax.sentences[0].span.end),
        )
        self.assertEqual(
            (0, 11),
            (self.syntax.paragraphs[0].span.begin, self.syntax.paragraphs[0].span.end),
        )
        self.assertEqual(
            (15, 21),
            (remapped_syntax.tokens[0].span.begin, remapped_syntax.tokens[0].span.end),
        )
        self.assertEqual(
            (25, 29),
            (remapped_syntax.tokens[1].span.begin, remapped_syntax.tokens[1].span.end),
        )
        self.assertEqual(
            (15, 29),
            (
                remapped_syntax.sentences[0].span.begin,
                remapped_syntax.sentences[0].span.end,
            ),
        )
        self.assertEqual("Sample <b>Text", remapped_syntax.sentences[0].span.text)
        self.assertEqual(
            (15, 29),
            (
                remapped_syntax.paragraphs[0].span.begin,
                remapped_syntax.paragraphs[0].span.end,
            ),
        )
        self.assertEqual("Sample <b>Text", remapped_syntax.paragraphs[0].span.text)

    def test_invalid_remap(self):
        with self.assertRaises(ValueError):
            self.html_doc.remap(
                dm.SyntaxPrediction(
                    text="IamOneGiantTokenWithManyCharacters",
                    producer_id=dm.ProducerId("Test", "1.0.0"),
                    tokens=[dm.Token(dm.Span(0, 34))],
                    sentences=[dm.Sentence((0, 34))],
                    paragraphs=[dm.Paragraph((0, 34))],
                )
            )


class TestSyntaxPrediction(TestCaseBase):
    def setUp(self):
        self.syntax = dm.SyntaxPrediction(
            text="Hello World!  I am HAL9000.",
            producer_id=dm.ProducerId("Test", "1.0.0"),
            tokens=[
                # note: tokens and sentences are assumed sorted, since annotations produced by
                # Izumo are sorted, and these objects are built by iterating on the output.
                # Reference: https://pages.github.ibm.com/ai-foundation/watson-nlp-documentation/changelog.html#0029-june-2021
                dm.Token(dm.Span(0, 5)),
                dm.Token(dm.Span(6, 11)),
                dm.Token(dm.Span(11, 12)),
                dm.Token(dm.Span(14, 15)),
                dm.Token(dm.Span(16, 18)),
                dm.Token(dm.Span(19, 26)),
                dm.Token(dm.Span(26, 27)),
            ],
            sentences=[dm.Sentence((0, 12)), dm.Sentence((14, 27))],
            paragraphs=[dm.Paragraph((0, 12)), dm.Paragraph((14, 27))],
        )

        self.syntax_minimal = dm.SyntaxPrediction("Hello World!")

        self.syntax_empty1 = dm.SyntaxPrediction("")
        self.syntax_empty2 = dm.SyntaxPrediction(" \n")
        self.syntax_empty3 = dm.SyntaxPrediction(
            text="", tokens=[], sentences=[], paragraphs=[]
        )

        self.syntax_empties = (
            self.syntax_empty1,
            self.syntax_empty2,
            self.syntax_empty3,
        )

    def test_fields(self):
        """Test that all fields in the protobuf are present."""
        self.assertTrue(utils.validate_fields(self.syntax))
        self.assertTrue(utils.validate_fields(self.syntax_minimal))
        for syntax_empty in self.syntax_empties:
            self.assertTrue(utils.validate_fields(syntax_empty))

    def test_from_proto_and_back(self):
        """Test that converting to protobuf and back results in the original data structure."""
        new = dm.SyntaxPrediction.from_proto(self.syntax.to_proto())
        self.assertEqual(new.text, self.syntax.text)
        self.assertEqual(new.producer_id.name, self.syntax.producer_id.name)
        self.assertEqual(new.tokens[0].span.begin, self.syntax.tokens[0].span.begin)
        self.assertEqual(
            new.sentences[0].span.begin, self.syntax.sentences[0].span.begin
        )
        self.assertEqual(
            new.paragraphs[0].span.begin, self.syntax.paragraphs[0].span.begin
        )

        new = dm.SyntaxPrediction.from_proto(self.syntax_minimal.to_proto())
        self.assertEqual(new.text, self.syntax_minimal.text)
        self.assertEqual(new.producer_id, None)
        self.assertEqual(len(new.tokens), 0)
        self.assertEqual(len(new.sentences), 0)
        self.assertEqual(len(new.paragraphs), 0)

    def test_from_json_and_back(self):
        """Test that converting to protobuf and back results in the original data structure."""
        new = dm.SyntaxPrediction.from_json(self.syntax.to_json())
        self.assertEqual(new.text, self.syntax.text)
        self.assertEqual(new.producer_id.name, self.syntax.producer_id.name)
        self.assertEqual(new.tokens[0].span.begin, self.syntax.tokens[0].span.begin)
        self.assertEqual(
            new.sentences[0].span.begin, self.syntax.sentences[0].span.begin
        )
        self.assertEqual(
            new.paragraphs[0].span.begin, self.syntax.paragraphs[0].span.begin
        )

        new = dm.SyntaxPrediction.from_json(self.syntax_minimal.to_json())
        self.assertEqual(new.text, self.syntax_minimal.text)
        self.assertEqual(new.producer_id, None)
        self.assertEqual(len(new.tokens), 0)
        self.assertEqual(len(new.sentences), 0)
        self.assertEqual(len(new.paragraphs), 0)

    def test_extract(self):
        """Test that text is correctly extracted in spans."""
        self.assertTrue(all(token.text for token in self.syntax.tokens))
        self.assertTrue(all(sentence.text for sentence in self.syntax.sentences))

        self.assertEqual(self.syntax.tokens[0].span.begin, 0)
        self.assertEqual(self.syntax.tokens[0].span.end, 5)
        self.assertEqual(self.syntax.tokens[-1].span.begin, 26)
        self.assertEqual(self.syntax.tokens[-1].span.end, 27)

        self.assertEqual(self.syntax.tokens[0].span.text, "Hello")
        self.assertEqual(self.syntax.tokens[0].span.text, "Hello")

        self.assertEqual(self.syntax.sentences[-1].span.text, "I am HAL9000.")
        self.assertEqual(self.syntax.sentences[-1].span.text, "I am HAL9000.")

    def test_get_token_texts(self):
        """Test that token texts are extracted correctly."""
        token_texts = self.syntax.get_token_texts()
        self.assertEqual(len(token_texts), len(self.syntax.tokens))
        self.assertEqual(len(token_texts), 7)
        self.assertTrue(all(isinstance(token_text, str) for token_text in token_texts))
        self.assertEqual(token_texts[0], "Hello")
        self.assertEqual(token_texts[-2], "HAL9000")
        self.assertEqual(token_texts[-1], ".")

        for syntax_empty in self.syntax_empties:
            token_texts = syntax_empty.get_token_texts()
            self.assertEqual(len(token_texts), 0)

    def test_get_POS_token_texts(self):
        """Test that POS tokens are extracted by token from a sentence correctly."""
        tokens = self.syntax.get_token_texts()
        pos_tokens = self.syntax.get_POS_texts()

        self.assertTrue(isinstance(pos_tokens, list))
        self.assertEqual(len(tokens), len(pos_tokens))
        self.assertEqual(len(pos_tokens), 7)

        for pos_token, token in zip(pos_tokens, self.syntax.tokens):
            self.assertEqual(
                pos_token, dm.enums.PartOfSpeechRev[token.part_of_speech.value]
            )

    def test_get_POS_token_texts_empty_syntax(self):
        """Test that POS tokens are extracted by token from an empty sentence
        correctly.
        """
        for syntax_empty in self.syntax_empties:
            pos_texts = syntax_empty.get_POS_texts()
            self.assertEqual(len(pos_texts), 0)

    def test_get_token_spans(self):
        """Test that token spans are extracted correctly."""
        token_spans = self.syntax.get_token_spans()
        self.assertEqual(len(token_spans), len(self.syntax.tokens))
        self.assertEqual(len(token_spans), 7)
        self.assertTrue(isinstance(token_span, tuple) for token_span in token_spans)
        self.assertEqual(token_spans[0], (0, 5))
        self.assertEqual(token_spans[-1], (26, 27))

        for syntax_empty in self.syntax_empties:
            token_spans = syntax_empty.get_token_spans()
            self.assertEqual(len(token_spans), 0)

    def test_get_tokens_by_sentence(self):
        """Test that tokens are extracted by sentence correctly."""
        tokens_by_sent = self.syntax.get_tokens_by_sentence()
        self.assertTrue(isinstance(tokens_by_sent, list))
        self.assertEqual(len(tokens_by_sent), len(self.syntax.sentences))
        self.assertEqual(len(tokens_by_sent), 2)
        self.assertEqual(len(tokens_by_sent[0]), 3)
        self.assertEqual(tokens_by_sent[0][0].span.text, "Hello")
        self.assertEqual(tokens_by_sent[0][0].span.begin, 0)
        self.assertEqual(tokens_by_sent[0][0].span.end, 5)
        self.assertEqual(tokens_by_sent[0][-1].span.text, "!")
        self.assertEqual(len(tokens_by_sent[-1]), 4)
        self.assertEqual(tokens_by_sent[-1][0].span.text, "I")
        self.assertEqual(tokens_by_sent[-1][-1].span.text, ".")
        self.assertEqual(tokens_by_sent[-1][-1].span.begin, 26)
        self.assertEqual(tokens_by_sent[-1][-1].span.end, 27)

        self.assertTrue(
            all(
                isinstance(token, dm.Token) for sent in tokens_by_sent for token in sent
            )
        )

        for syntax_empty in self.syntax_empties:
            tokens_by_sent = syntax_empty.get_tokens_by_sentence()
            self.assertTrue(isinstance(tokens_by_sent, list))
            self.assertEqual(len(tokens_by_sent[0]), 0)

    def test_get_tokens_by_sentence_invalid(self):
        """Test that exceptions are thrown when attempting to extract tokens by sentence when
        sentences or tokens are malformed, i.e., don't align.
        """
        syntax_missing_sentence = dm.SyntaxPrediction(
            text="Hello World!  I am HAL9000.",
            tokens=[
                dm.Token(dm.Span(6, 11)),
                dm.Token(dm.Span(0, 5)),
                dm.Token(dm.Span(11, 12)),
                dm.Token(dm.Span(14, 15)),
                dm.Token(dm.Span(16, 18)),
                dm.Token(dm.Span(26, 27)),
                dm.Token(dm.Span(19, 26)),
            ],
            sentences=[dm.Sentence((0, 12))],
        )  # no second sentence!

        with self.assertRaises(ValueError):
            syntax_missing_sentence.get_tokens_by_sentence()

        syntax_token_overlaps_sentences = dm.SyntaxPrediction(
            text="Hello World!  I am HAL9000.",
            tokens=[
                dm.Token(dm.Span(6, 11)),
                dm.Token(dm.Span(0, 5)),
                dm.Token(dm.Span(11, 18)),  # token spans both sentences!
                dm.Token(dm.Span(14, 15)),
                dm.Token(dm.Span(16, 18)),
                dm.Token(dm.Span(26, 27)),
                dm.Token(dm.Span(19, 26)),
            ],
            sentences=[dm.Sentence((0, 12)), dm.Sentence((14, 27))],
        )

        with self.assertRaises(ValueError):
            syntax_token_overlaps_sentences.get_tokens_by_sentence()

    def test_get_tokens_by_sentence_max_len(self):
        """Test that the max_sent_toks argument correctly splits long sentences."""
        syntax_long_sentence = dm.SyntaxPrediction(
            text="This is a long sentence that needs to be split into smaller sentences.  "
            "This one is short.",
            tokens=[
                dm.Token(dm.Span(0, 4)),
                dm.Token(dm.Span(5, 7)),
                dm.Token(dm.Span(8, 9)),
                dm.Token(dm.Span(10, 14)),
                dm.Token(dm.Span(15, 23)),
                dm.Token(dm.Span(24, 28)),
                dm.Token(dm.Span(29, 34)),
                dm.Token(dm.Span(35, 37)),
                dm.Token(dm.Span(38, 40)),
                dm.Token(dm.Span(41, 46)),
                dm.Token(dm.Span(47, 51)),
                dm.Token(dm.Span(52, 59)),
                dm.Token(dm.Span(60, 69)),
                dm.Token(dm.Span(69, 70)),
                dm.Token(dm.Span(72, 76)),
                dm.Token(dm.Span(77, 80)),
                dm.Token(dm.Span(81, 83)),
                dm.Token(dm.Span(84, 89)),
                dm.Token(dm.Span(89, 90)),
            ],
            sentences=[dm.Sentence((0, 70)), dm.Sentence((72, 90))],
        )

        self.assertEqual(len(syntax_long_sentence.tokens), 19)
        self.assertEqual(len(syntax_long_sentence.sentences), 2)

        tokens_by_sentence = syntax_long_sentence.get_tokens_by_sentence(
            max_sent_toks=100
        )
        self.assertEqual(len(tokens_by_sentence), len(syntax_long_sentence.sentences))

        tokens_by_sentence = syntax_long_sentence.get_tokens_by_sentence(
            max_sent_toks=None
        )
        self.assertEqual(len(tokens_by_sentence), len(syntax_long_sentence.sentences))

        tokens_by_sentence = syntax_long_sentence.get_tokens_by_sentence(
            max_sent_toks=5
        )
        self.assertEqual(len(tokens_by_sentence), 4)
        self.assertEqual(tokens_by_sentence[0][0].span.text, "This")
        self.assertEqual(tokens_by_sentence[1][0].span.text, "that")
        self.assertEqual(tokens_by_sentence[2][0].span.text, "into")

        for max_sent_toks in (2, 4, 7, 100):
            tokens_by_sentence = syntax_long_sentence.get_tokens_by_sentence(
                max_sent_toks=max_sent_toks
            )

            self.assertGreaterEqual(
                len(tokens_by_sentence), len(syntax_long_sentence.sentences)
            )

            for sentence in tokens_by_sentence:
                self.assertGreater(len(sentence), 0)
                self.assertLessEqual(len(sentence), max_sent_toks)

    def test_get_tokens_by_span(self):
        """Test that the function returns all the tokens inside a valid span."""
        tokens = self.syntax.get_tokens_by_span(dm.Span(0, 11))

        self.assertEqual(tokens[0].span.begin, 0)
        self.assertEqual(tokens[0].span.end, 5)
        self.assertEqual(tokens[1].span.begin, 6)
        self.assertEqual(tokens[1].span.end, 11)

    def test_get_tokens_by_span_partial(self):
        """Test that the function returns the tokens inside a partial span."""
        tokens = self.syntax.get_tokens_by_span(dm.Span(0, 9))

        self.assertEqual(tokens[0].span.begin, 0)
        self.assertEqual(tokens[0].span.end, 5)
        self.assertEqual(len(tokens), 1)

    def test_get_tokens_by_span_empty(self):
        """Test that the function returns nothing for out of range span."""
        tokens = self.syntax.get_tokens_by_span(dm.Span(1000, 1100))
        self.assertFalse(tokens)

    def test_find_token(self):
        """Test that the function returns token for valid begin offset."""
        index = self.syntax.find_token(6)

        self.assertEqual(self.syntax.tokens[index].span.begin, 6)
        self.assertEqual(self.syntax.tokens[index].span.end, 11)

    def test_find_token_invalid_begin_offset(self):
        """Test that the function returns -1 for invalid begin offset."""
        index = self.syntax.find_token(1)
        self.assertEqual(index, -1)

    def test_find_token_out_of_range_offset(self):
        """Test that the function returns -1 for out of range begin offset."""
        index = self.syntax.find_token(1000)
        self.assertEqual(index, -1)

    def test_get_sentence_containing_span(self):
        """Test that the function returns sentence for valid span and None otherwise"""
        # Searching for valid sentence: dm.Sentence((0, 12))
        expected_sent = (self.syntax.sentences[0], False)
        self.assertEqual(
            expected_sent, self.syntax.get_sentence_containing_span(dm.Span(0, 2))
        )
        self.assertEqual(
            expected_sent, self.syntax.get_sentence_containing_span(dm.Span(0, 12))
        )
        self.assertEqual(
            expected_sent, self.syntax.get_sentence_containing_span(dm.Span(4, 5))
        )
        self.assertEqual(
            expected_sent, self.syntax.get_sentence_containing_span(dm.Span(4, 12))
        )

        # overlapping span in multiple sentences
        sentence, overlap_found = self.syntax.get_sentence_containing_span(
            dm.Span(4, 14)
        )
        self.assertIsNotNone(sentence)
        self.assertTrue(overlap_found)

        # span given is outside of any of the sentences - no valid sentence found
        sentence, overlap_found = self.syntax.get_sentence_containing_span(
            dm.Span(1000, 1004)
        )
        self.assertIsNone(sentence)

    def test_tokens_convert_tuples_to_spans(self):
        """Ensure that we can create a token from a tuple (this functionality is deprecated)."""
        tok = dm.Token((0, 1))
        self.assertIsInstance(tok, dm.Token)
        self.assertIsInstance(tok.span, dm.Span)
        self.assertEqual(tok.span.begin, 0)
        self.assertEqual(tok.span.end, 1)
