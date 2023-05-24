# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#
# Standard
import os

# Third Party
import numpy as np
import watson_nlp

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase

## Constants ###################################################################


class TestEmbeddingPrediction(TestCaseBase):
    """Tests for Embedding Predictions. EmbeddingPrediction objects are texts that have embedded,
    i.e., the output of Embedding block types. All Embedding algorithms produce EmbeddingPrediction
    objects.
    """

    def setUp(self):
        self.numpy_2d = np.array(range(16), dtype=np.float32).reshape((4, 4))
        self.sample_densemat = dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2)
        self.vocab_to_idx = {
            "<pad>": 0,
            "<unk>": 1,
            "embed": 2,
            "embedding": 3,
            "vector": 4,
            "vectorization": 5,
        }
        self.emb_pred_with_offsets = dm.EmbeddingPrediction(
            data=self.numpy_2d,
            offsets=(0, 3),
            producer_id=dm.ProducerId("Test", "1.2.3"),
        )
        self.emb_pred_without_offsets = dm.EmbeddingPrediction(
            data=self.numpy_2d, producer_id=dm.ProducerId("Test", "1.2.3")
        )
        self.embed = dm.Embedding(
            data=self.numpy_2d,
            vocab_to_idx=self.vocab_to_idx,
            producer_id=dm.ProducerId("EmbedTest", "1.2.3"),
        )
        self.tokens = ["its", "very", "windy", "today"]

    def test_from_proto_and_back(self):
        """Test that Embedding Predictions can go to / from proto."""
        # NOTE: Data type for embeddings is standardized to float32
        new_pred = dm.EmbeddingPrediction.from_proto(
            self.emb_pred_with_offsets.to_proto()
        )
        self.assertTrue(isinstance(new_pred.data, np.ndarray))
        self.assertTrue(np.all(new_pred.data == self.numpy_2d))

    def test_from_json_and_back(self):
        """Test that Embedding Predictions can go to / from json."""
        # NOTE: Data type for embeddings is standardized to float32
        new_pred = dm.EmbeddingPrediction.from_json(
            self.emb_pred_with_offsets.to_json()
        )
        self.assertTrue(isinstance(new_pred.data, np.ndarray))
        self.assertTrue(np.all(new_pred.data == self.numpy_2d))

    def test_init_fails_with_dense_matrix(self):
        """Test that Embedding Predictions may only be constructed with numpy arrays."""
        self.assertRaises(TypeError, dm.EmbeddingPrediction, self.sample_densemat)

    def test_init_casts_numpy_arrays_to_float32(self):
        """Test that initializing EmbeddingPredictions casts numpy arrays to np.float32."""
        int_arr = np.array(1, dtype=np.int64)
        new_pred = dm.EmbeddingPrediction(data=int_arr)
        self.assertEqual(new_pred.data.dtype, np.float32)

    def test_embedding_from_proto_and_back(self):
        """Test that Embedding can go to / from proto."""
        # NOTE: Data type for embeddings is standardized to float32
        new_embed = dm.Embedding.from_proto(self.embed.to_proto())
        vocab_dict = dict(new_embed.vocab_to_idx)
        self.assertTrue(isinstance(new_embed.data, np.ndarray))
        self.assertTrue(isinstance(vocab_dict, dict))
        self.assertEqual(vocab_dict["vector"], 4)
        self.assertEqual(new_embed.pad_word, self.embed.pad_word)
        self.assertEqual(new_embed.unk_word, self.embed.unk_word)

    def test_embedding_from_json_and_back(self):
        """Test that Embeddings can go to / from json."""
        # NOTE: Data type for embeddings is standardized to float32
        new_embed = dm.Embedding.from_json(self.embed.to_json())
        vocab_dict = dict(new_embed.vocab_to_idx)
        self.assertTrue(isinstance(new_embed.data, np.ndarray))
        self.assertTrue(isinstance(vocab_dict, dict))
        self.assertTrue(np.all(new_embed.data == self.numpy_2d))
        self.assertEqual(vocab_dict["vectorization"], 5)
        self.assertEqual(new_embed.pad_word, self.embed.pad_word)
        self.assertEqual(new_embed.unk_word, self.embed.unk_word)

    def test_embedding_gets_default_values_for_pad_and_unk_words(self):
        """Test that Embedding gets default values <pad> and <unk> for pad_word and unk_word respectively."""
        new_embed = dm.Embedding(self.numpy_2d, self.vocab_to_idx)
        self.assertEqual(new_embed.pad_word, "<pad>")
        self.assertEqual(new_embed.unk_word, "<unk>")

    def test_embedding_overrides_with_default_values_for_empty_pad_and_unk_words(self):
        """Test that Embedding gets default values <pad> and <unk> for pad_word and unk_word respectively,
        if a protobuf loads embedding with no pad_word and unk_word set."""
        new_embed = dm.Embedding.from_proto(
            dm.Embedding(self.numpy_2d, self.vocab_to_idx, "", "").to_proto()
        )
        self.assertEqual(new_embed.pad_word, "<pad>")
        self.assertEqual(new_embed.unk_word, "<unk>")

    def test_get_padded_token_indices_function_takes_longer_documents(self):
        """Make sure get_padded_token_indices function works for a document
        longer than the max_len
        """
        max_len = len(self.tokens) - 1
        result = self.embed.get_padded_token_indices(self.tokens, max_len)
        self.assertEqual(result.shape[0], max_len)

    def test_get_padded_token_indices_function_pads_shorter_docs(self):
        """Make sure get_padded_token_indices function pads documents that are shorter than max len"""
        max_len = len(self.tokens) + 1
        result = self.embed.get_padded_token_indices(self.tokens, max_len)
        self.assertEqual(result.shape[0], max_len)
        self.assertEqual(result[-1], 0)

    def test_embedding_vocab_order(self):
        """Make sure the order is kept in vocab retrieval, regardless of dict init order."""
        vocab_to_idx = {}
        # NOTE: This test is significant because Python 3.6+ gives keys in
        # insertion order if you just call keys() directly on the dictionary.
        vocab_to_idx["<unk>"] = 1
        vocab_to_idx["<pad>"] = 0
        embed = dm.Embedding(
            data=self.numpy_2d,
            vocab_to_idx=vocab_to_idx,
            producer_id=dm.ProducerId("EmbedTest", "1.2.3"),
        )
        ordered_vocab = embed.get_ordered_vocab()
        self.assertTrue(isinstance(ordered_vocab, list))
        self.assertEqual(len(ordered_vocab), 2)
        self.assertEqual(ordered_vocab, ["<pad>", "<unk>"])

    def test_disassemble_with_offsets(self):
        """Ensure that if we .disassemble() something with valid offsets, it works properly."""
        disassembled_preds = self.emb_pred_with_offsets.disassemble()
        rows_counted = 0
        # Disassembling is only allowed if our pred represents multiple input source objects,
        # otherwise it would just return a list containing the original object, which is silly.
        self.assertTrue(len(disassembled_preds) > 1)
        for pred in disassembled_preds:
            # Get the same number of sentences as in our disassembled predictions and make
            # sure our alignment is working properly here by checking all values.
            end_offset = pred.data.shape[0] + rows_counted
            submat = self.emb_pred_with_offsets.data[rows_counted:end_offset, :]
            self.assertTrue(np.allclose(pred.data, submat.data))
            # Move the starting index for the next check.
            rows_counted += pred.data.shape[0]

    def test_disassemble_without_offsets(self):
        """Ensure that if we .disassemble() something without offsets, we raise (nothing to do)."""
        with self.assertRaises(ValueError):
            self.emb_pred_without_offsets.disassemble()

    def test_offset_standardization_for_none_cases(self):
        """Test that offsets representing one document correctly map to offsets=None."""
        inputs_mapping_to_none = [[], [0], None]
        for inp in inputs_mapping_to_none:
            pred = dm.EmbeddingPrediction(data=self.numpy_2d, offsets=inp)
            self.assertEqual(pred.offsets, None)

    def test_out_of_bounds_offsets_raise(self):
        """Test offsets may not be out of bounds under any circumstances."""
        out_of_bounds_inputs = [[-1], [100]]
        for inp in out_of_bounds_inputs:
            with self.assertRaises(ValueError):
                dm.EmbeddingPrediction(data=self.numpy_2d, offsets=inp)

    def test_offsets_must_be_integers(self):
        """Test offsets must be integers."""
        with self.assertRaises(TypeError):
            dm.EmbeddingPrediction(data=self.numpy_2d, offsets=[0, 1.4])

    def test_offsets_must_be_unique(self):
        """Test must be unique values."""
        with self.assertRaises(ValueError):
            dm.EmbeddingPrediction(data=self.numpy_2d, offsets=[0, 0, 0])

    def test_offsets_are_sorted(self):
        """Test that offsets are sorted when building the object (needed for disassembling)."""
        pred = dm.EmbeddingPrediction(data=self.numpy_2d, offsets=[2, 0])
        self.assertEqual(pred.offsets, [0, 2])

    def test_offset_standarization_for_zero(self):
        """Test that 0 is included in offsets if it's not provided."""
        without_zero_pred = dm.EmbeddingPrediction(data=self.numpy_2d, offsets=[2])
        self.assertEqual(without_zero_pred.offsets[0], 0)
        with_zero_pred = dm.EmbeddingPrediction(data=self.numpy_2d, offsets=[0, 2])
        self.assertEqual(without_zero_pred.offsets, with_zero_pred.offsets)
