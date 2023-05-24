# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Third Party
from scipy.sparse import csr_matrix
import numpy as np
import utils

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestVectorizationPrediction(TestCaseBase):
    """Tests for Vectorization Predictions. VectorizationPrediction objects are texts that have
    embedded, i.e., the output of Vectorization block types. All Vectorization algorithms produce
    VectorizationPrediction objects.
    """

    def setUp(self):
        # Like before, create a sparse matrix in scipy & and equivalent SparseMatrix
        self.scipy_sparsemat = csr_matrix(np.eye(2))
        self.scipy_sparsemat_int32 = self.scipy_sparsemat.astype(
            dtype=np.dtype("int32")
        )
        self.scipy_sparsemat_float64 = self.scipy_sparsemat.astype(
            dtype=np.dtype("float64")
        )
        self.sample_sparsemat = dm.SparseMatrix(
            data=[1, 1], indices=[0, 1], indptr=[0, 1, 2], rows=2, cols=2
        )
        self.vocab_to_idx = {"vector": 0, "vectorization": 1}
        # Then wrap the Scipy matrix in a VectorizationPrediction
        self.vec_pred = dm.VectorizationPrediction(
            data=self.scipy_sparsemat, producer_id=dm.ProducerId("Test", "1.2.3")
        )
        self.vec = dm.Vectorization(
            data=self.scipy_sparsemat,
            vocab_to_idx=self.vocab_to_idx,
            producer_id=dm.ProducerId("VectorTest", "1.2.4"),
        )

        self.vec_pred_int32 = dm.VectorizationPrediction(
            data=self.scipy_sparsemat,
            dtype=np.dtype("int32"),
            producer_id=dm.ProducerId("Test", "1.2.3"),
        )

        self.vec_float64 = dm.Vectorization(
            data=self.scipy_sparsemat,
            dtype=np.dtype("float64"),
            vocab_to_idx=self.vocab_to_idx,
            producer_id=dm.ProducerId("VectorTest", "1.2.4"),
        )

        self.vec_pred_keep_dtype_int32 = dm.VectorizationPrediction(
            data=self.scipy_sparsemat_int32,
            keep_dtype=True,
            producer_id=dm.ProducerId("Test", "1.2.3"),
        )
        self.vec_keep_dtype_float64 = dm.Vectorization(
            data=self.scipy_sparsemat_float64,
            keep_dtype=True,
            vocab_to_idx=self.vocab_to_idx,
            producer_id=dm.ProducerId("VectorTest", "1.2.4"),
        )

        self.vec_pred_keep_dtype_priority = dm.VectorizationPrediction(
            data=self.scipy_sparsemat,
            dtype=np.dtype("int32"),
            keep_dtype=True,
            producer_id=dm.ProducerId("VectorTest", "1.2.4"),
        )
        self.vec_keep_dtype_priority = dm.Vectorization(
            data=self.scipy_sparsemat,
            dtype=np.dtype("float64"),
            keep_dtype=True,
            vocab_to_idx=self.vocab_to_idx,
            producer_id=dm.ProducerId("VectorTest", "1.2.4"),
        )

    def test_from_proto_and_back(self):
        """Test that Vectorization Predictions can go to / from proto."""
        # NOTE: Data type for vectorizations is standardized to float32 by default
        new_pred = dm.VectorizationPrediction.from_proto(self.vec_pred.to_proto())
        self.assertTrue(isinstance(new_pred.data, csr_matrix))
        # Ensure that given sizing information, the start/end csr matrices are equivalent
        self.assertTrue((new_pred.data != self.scipy_sparsemat).nnz == 0)

    def test_from_json_and_back(self):
        """Test that Vectorization Predictions can go to / from json."""
        # NOTE: Data type for vectorizations is standardized to float32 by default
        new_pred = dm.VectorizationPrediction.from_json(self.vec_pred.to_json())
        self.assertTrue(isinstance(new_pred.data, csr_matrix))
        # Ensure that given sizing information, the start/end csr matrices are equivalent
        self.assertTrue((new_pred.data != self.scipy_sparsemat).nnz == 0)

    def test_init_fails_with_sparse_matrix(self):
        """Test that Vectorization Predictions may only be constructed with Scipy CSR matrices."""
        self.assertRaises(TypeError, dm.EmbeddingPrediction, self.sample_sparsemat)

    def test_init_casts_scipy_matrices_to_float32(self):
        """Test that initializing VectorizationPredictions casts scipy arrays to np.float32 by default."""
        int_arr = csr_matrix(np.array(1, dtype=np.int64))
        new_pred = dm.VectorizationPrediction(data=int_arr)
        self.assertEqual(new_pred.data.dtype, np.float32)

    def test_vector_from_proto_and_back(self):
        """Test that Vectorization can go to / from proto."""
        # NOTE: Data type for vectorizations is standardized to float32 by default
        new_pred = dm.Vectorization.from_proto(self.vec.to_proto())
        vocab_dict = dict(new_pred.vocab_to_idx)
        self.assertTrue(isinstance(new_pred.data, csr_matrix))
        # Ensure that given sizing information, the start/end csr matrices are equivalent
        self.assertTrue((new_pred.data != self.scipy_sparsemat).nnz == 0)
        self.assertTrue(isinstance(vocab_dict, dict))
        self.assertEquals(vocab_dict["vector"], 0)

    def test_vector_from_json_and_back(self):
        """Test that Vectorization can go to / from json."""
        # NOTE: Data type for vectorizations is standardized to float32 by default
        new_pred = dm.Vectorization.from_json(self.vec.to_json())
        vocab_dict = dict(new_pred.vocab_to_idx)
        self.assertTrue(isinstance(new_pred.data, csr_matrix))
        # Ensure that given sizing information, the start/end csr matrices are equivalent
        self.assertTrue((new_pred.data != self.scipy_sparsemat).nnz == 0)
        self.assertEquals(vocab_dict["vectorization"], 1)

    def test_vectorization_vocab_order(self):
        """Make sure the order is kept in vocab retrieval, regardless of dict init order."""
        vocab_to_idx = {}
        # NOTE: This test is significant because Python 3.6+ gives keys in
        # insertion order if you just call keys() directly on the dictionary.
        vocab_to_idx["<unk>"] = 1
        vocab_to_idx["<pad>"] = 0
        vec = dm.Vectorization(
            data=self.scipy_sparsemat,
            vocab_to_idx=vocab_to_idx,
            producer_id=dm.ProducerId("VectorTest", "1.2.3"),
        )
        ordered_vocab = vec.get_ordered_vocab()
        self.assertTrue(isinstance(ordered_vocab, list))
        self.assertEqual(len(ordered_vocab), 2)
        self.assertEqual(ordered_vocab, ["<pad>", "<unk>"])

    def test_default_dtype(self):
        # the default dtype is float32
        self.assertEqual(self.vec_pred.data.dtype, np.dtype("float32"))

    def test_set_dtype(self):
        # if we specify a custom dtype it should be used
        self.assertEqual(self.vec_pred_int32.data.dtype, np.dtype("int32"))

    def test_keep_dtype(self):
        # if we request to keep the type, it should be preserved
        self.assertEqual(self.vec_pred_keep_dtype_int32.data.dtype, np.dtype("int32"))

    def test_keep_and_set_dtype(self):
        # keep_dtype takes precedence over setting the dtype
        self.assertEqual(
            self.vec_pred_keep_dtype_priority.data.dtype,
            self.scipy_sparsemat.data.dtype,
        )

    def test_invalid_keep_dtype(self):
        with self.assertRaises(TypeError):
            dm.VectorizationPrediction(
                data=csr_matrix(np.eye(2)),
                keep_dtype="yes",
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_invalid_set_dtype(self):
        with self.assertRaises(TypeError):
            dm.VectorizationPrediction(
                data=csr_matrix(np.eye(2)),
                dtype="int32",
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_vector_default_dtype(self):
        # the default dtype is float32
        self.assertEqual(self.vec.data.dtype, np.dtype("float32"))

    def test_vector_set_dtype(self):
        # if we specify a custom dtype it should be used
        self.assertEqual(self.vec_float64.data.dtype, np.dtype("float64"))

    def test_vector_keep_dtype(self):
        # if we request to keep the type, it should be preserved
        self.assertEqual(self.vec_keep_dtype_float64.data.dtype, np.dtype("float64"))

    def test_vector_keep_set_dtype(self):
        # keep_dtype takes precedence over setting the dtype
        self.assertEqual(
            self.vec_keep_dtype_priority.data.dtype, self.scipy_sparsemat.data.dtype
        )

    def test_vector_invalid_keep_dtype(self):
        with self.assertRaises(TypeError):
            dm.Vectorization(
                data=csr_matrix(np.eye(2)),
                keep_dtype="yes",
                vocab_to_idx=self.vocab_to_idx,
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_vector_invalid_dtype(self):
        with self.assertRaises(TypeError):
            dm.Vectorization(
                data=csr_matrix(np.eye(2)),
                dtype="int32",
                vocab_to_idx=self.vocab_to_idx,
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_from_proto_and_back_custom_dtype(self):
        """Test that Vectorization Predictions preserves a custom dtype when go to / from proto."""
        new_pred = dm.VectorizationPrediction.from_proto(self.vec_pred_int32.to_proto())
        self.assertEqual(new_pred.data.dtype, self.vec_pred_int32.data.dtype)

    def test_from_json_and_back_custom_dtype(self):
        """Test that Vectorization Predictions preserves a custom dtype when go to / from json."""
        new_pred = dm.VectorizationPrediction.from_json(self.vec_pred_int32.to_json())
        self.assertEqual(new_pred.data.dtype, self.vec_pred_int32.data.dtype)

    def test_vector_from_proto_and_back_custom_dtype(self):
        """Test that Vectorization preserves a custom dtype when go to / from proto and ."""
        new_pred = dm.Vectorization.from_proto(self.vec_float64.to_proto())
        self.assertEqual(new_pred.data.dtype, self.vec_float64.data.dtype)

    def test_vector_from_json_and_back_custom_dtype(self):
        """Test that Vectorization preserves a custom dtype when go to / from json."""
        new_pred = dm.Vectorization.from_json(self.vec_float64.to_json())
        self.assertEqual(new_pred.data.dtype, self.vec_float64.data.dtype)
