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

# Third Party
from scipy import sparse
import numpy as np

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestDenseMatrix(TestCaseBase):
    """Tests for DenseMatrix. Dense matrices are generally used for transmission and display of
    dense embeddings, since numpy arrays don't play nicely out of the box with proto / json
    serialization.
    """

    def setUp(self):
        self.numpy_1d = np.array(range(16))
        self.numpy_2d = self.numpy_1d.reshape((4, 4))
        self.numpy_3d = self.numpy_1d.reshape((2, 2, 4))
        self.sample_densemat = dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2)

    # [Successful creation tests not covered by setup]
    def test_empty_dense_mat_construction(self):
        """Test that an empty matrix can be created with 0/0 rows and columns"""
        densemat = dm.DenseMatrix(data=[], rows=0, cols=0)
        self.assertEqual(densemat.rows, 0)
        self.assertEqual(densemat.rows, 0)
        self.assertListEqual(densemat.data, [])

    def test_construction_from_2d_numpy_array_succeeds(self):
        """Test that DenseMatrix objects can be constructed from 2D numpy matrices."""
        densemat = dm.DenseMatrix.from_numpy_array(self.numpy_2d)
        # shape info represented solely by rows/cols - data is flattened and should be unrolled to
        # a tuple of flat data, which should match the 1D array when converted back to numpy
        erows, ecols = self.numpy_2d.shape
        self.assertEqual(densemat.rows, erows)
        self.assertEqual(densemat.rows, ecols)
        self.assertTrue(np.all(np.array(densemat.data) == self.numpy_1d))

    # [Failure creation tests]
    def test_dense_mat_construction_fails_with_misaligned_shape_and_data(self):
        """Test that construction fails if the rows and cols can't match the data given."""
        self.assertRaises(ValueError, dm.DenseMatrix, data=[1, 2, 3, 4], rows=1, cols=2)

    def test_dense_mat_construction_fails_with_negative_row_col_values(self):
        """Test that construction fails if negative row/col values are provided."""
        self.assertRaises(
            ValueError, dm.DenseMatrix, data=[1, 2, 3, 4], rows=-2, cols=-2
        )

    def test_construction_from_non_2d_numpy_array_fails(self):
        """Test that DenseMatrix objects fail for numpy arrays that are not 2D."""
        self.assertRaises(ValueError, dm.DenseMatrix.from_numpy_array, self.numpy_1d)
        self.assertRaises(ValueError, dm.DenseMatrix.from_numpy_array, self.numpy_3d)

    # [Conversion Tests]
    def test_from_proto_and_back(self):
        """Test that a dense matrix can be converted to proto and back."""
        new_mat = dm.DenseMatrix.from_proto(self.sample_densemat.to_proto())
        self.assertListEqual(new_mat.data, self.sample_densemat.data)

    def test_from_json_and_back(self):
        """Test that a dense matrix can be converted to json and back."""
        new_mat = dm.DenseMatrix.from_json(self.sample_densemat.to_json())
        self.assertListEqual(new_mat.data, self.sample_densemat.data)

    def test_fields(self):
        """Test that fields on the DenseMatrix object are valid."""
        self.assertTrue(self.validate_fields(self.sample_densemat))

    def test_emb_stack_vectors(self):
        """Test that we can stack encapsulated EmbeddingPrediction data vectors."""
        sample_data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        emb_preds = [
            dm.EmbeddingPrediction(np.array(vec).reshape(1, -1)) for vec in sample_data
        ]
        stacked_mat = dm.DenseMatrix.stack_raw_tensor(emb_preds)
        self.assertIsInstance(stacked_mat, np.ndarray)
        self.assertEqual(stacked_mat.shape[0], 2)
        for idx, vec in enumerate(sample_data):
            self.assertEqual(vec, stacked_mat[idx].tolist())

    def test_emb_stack_matrices_fails_when_disallowed(self):
        """Test that we explode when stacking matrices if we explicitly disallow matrices."""
        sample_data = [[1, 2, 3, 4], [[5, 6, 7, 8], [9, 10, 11, 12]]]
        emb_preds = [
            dm.EmbeddingPrediction(np.array(vec).reshape(-1, 4)) for vec in sample_data
        ]
        with self.assertRaises(ValueError):
            dm.DenseMatrix.stack_raw_tensor(emb_preds, allow_matrices=False)

    def test_emb_stack_matrices_succeeds_when_allowed(self):
        """Test that we explode when stacking matrices if we explicitly allow matrices."""
        sample_data = [[1, 2, 3, 4], [[5, 6, 7, 8], [9, 10, 11, 12]]]
        # Expected flat result after nesting is considered and done
        expected_data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        emb_preds = [
            dm.EmbeddingPrediction(np.array(vec).reshape(-1, 4)) for vec in sample_data
        ]
        stacked_mat = dm.DenseMatrix.stack_raw_tensor(emb_preds, allow_matrices=True)
        self.assertIsInstance(stacked_mat, np.ndarray)
        # Stacking a (1 x n) + (2 x n) should give us a (3 x n)
        self.assertEqual(stacked_mat.shape[0], 3)
        for idx, vec in enumerate(expected_data):
            self.assertEqual(vec, stacked_mat[idx].tolist())

    def test_invalid_dtype_str(self):
        with self.assertRaises(TypeError):
            dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype="float642")

    def test_invalid_dtype_type(self):
        with self.assertRaises(TypeError):
            dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype=int)

    def test_default_matrix_dtype(self):
        self.assertEqual(
            np.dtype(dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2).dtype),
            np.dtype("float32"),
        )

    def test_int32_matrix(self):
        self.assertEqual(
            np.dtype(
                dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype="int32").dtype
            ),
            np.dtype("int32"),
        )

    def test_float64_matrix(self):
        self.assertEqual(
            np.dtype(
                dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype="float64").dtype
            ),
            np.dtype("float64"),
        )

    def test_int64_matrix(self):
        self.assertEqual(
            np.dtype(
                dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype="int64").dtype
            ),
            np.dtype("int64"),
        )

    def test_from_numpy_int32(self):
        self.assertEqual(
            np.dtype(
                dm.DenseMatrix.from_numpy_array(
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.dtype("int32"))
                ).dtype
            ),
            np.dtype("int32"),
        )

    def test_to_numpy_int32(self):
        self.assertEqual(
            dm.DenseMatrix.to_numpy_array(
                dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype="int32")
            ).dtype,
            np.dtype("int32"),
        )

    def test_to_dict_int32(self):
        self.assertEqual(
            dm.DenseMatrix.to_dict(
                dm.DenseMatrix(data=[1, 2, 3, 4], rows=2, cols=2, dtype="int32")
            )["dtype"],
            "int32",
        )


class TestSparseMatrix(TestCaseBase):
    """Tests for SparseMatrix. Sparse matrices are generally used for transmission and display of
    sparse vectorizations, since scipy arrays don't play nicely out of the box with proto / json
    serialization.
    """

    def setUp(self):
        self.scipy_sparsemat = sparse.csr_matrix(np.eye(2))
        self.sample_data = [1, 1]
        self.sample_indices = [0, 1]
        self.sample_indptr = [0, 1, 2]
        self.sample_rows, self.sample_cols = (2, 2)
        self.sample_sparsemat = dm.SparseMatrix(
            data=self.sample_data,
            indices=self.sample_indices,
            indptr=self.sample_indptr,
            rows=self.sample_rows,
            cols=self.sample_cols,
        )

    # [Successful creation tests not covered by setup]
    def test_empty_sparse_mat_construction(self):
        """Test that an empty sparse matrix can be created with 0/0 rows and columns."""
        sparsemat = dm.SparseMatrix(data=[], indices=[], indptr=[0], rows=0, cols=0)
        self.assertEqual(sparsemat.rows, 0)
        self.assertEqual(sparsemat.rows, 0)
        self.assertListEqual(sparsemat.data, [])
        self.assertListEqual(sparsemat.indices, [])
        self.assertListEqual(sparsemat.indptr, [0])

    def test_construction_from_scipy_csr_matrix_succeeds(self):
        """Test that SparseMatrix objects can be constructed from scipy csr matrices."""
        sparsemat = dm.SparseMatrix.from_scipy_csr_matrix(self.scipy_sparsemat)
        self.assertEqual(sparsemat.rows, self.sample_rows)
        self.assertEqual(sparsemat.rows, self.sample_cols)
        self.assertListEqual(sparsemat.data, self.sample_data)
        self.assertListEqual(sparsemat.indices, self.sample_indices)
        self.assertListEqual(sparsemat.indptr, self.sample_indptr)

    # [Failure creation tests]
    def test_sparse_mat_construction_fails_with_more_data_vals_than_allowed(self):
        """Test that construction fails if we provide more values than possible for the shape"""
        self.assertRaises(
            ValueError,
            dm.SparseMatrix,
            data=self.sample_data,
            indices=self.sample_indices,
            indptr=self.sample_indptr,
            rows=1,
            cols=1,
        )

    def test_sparse_mat_construction_fails_with_negative_row_col_values(self):
        """Test that construction fails if negative row/col values are provided."""
        self.assertRaises(
            ValueError,
            dm.SparseMatrix,
            data=self.sample_data,
            indices=self.sample_indices,
            indptr=self.sample_indptr,
            rows=-self.sample_rows,
            cols=-self.sample_cols,
        )

    # [Conversion Tests]
    def test_from_proto_and_back(self):
        """Test that a sparse matrix can be converted to proto and back."""
        new_mat = dm.SparseMatrix.from_proto(self.sample_sparsemat.to_proto())
        self.assertListEqual(new_mat.data, self.sample_sparsemat.data)
        self.assertListEqual(new_mat.indices, self.sample_sparsemat.indices)
        self.assertListEqual(new_mat.indptr, self.sample_sparsemat.indptr)

    def test_from_json_and_back(self):
        """Test that a sparse matrix can be converted to json and back."""
        new_mat = dm.SparseMatrix.from_json(self.sample_sparsemat.to_json())
        self.assertEqual(new_mat.data, self.sample_sparsemat.data)

    def test_fields(self):
        """Test that fields on the SparseMatrix object are valid."""
        self.assertTrue(self.validate_fields(self.sample_sparsemat))

    def test_vectorization_stack_vectors(self):
        """Test that we can stack encapsulated VectorizationPrediction data vectors."""
        sample_data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        vec_preds = [
            dm.VectorizationPrediction(sparse.csr_matrix(vec)) for vec in sample_data
        ]
        stacked_mat = dm.SparseMatrix.stack_raw_tensor(vec_preds)
        self.assertIsInstance(stacked_mat, sparse.csr_matrix)
        self.assertEqual(stacked_mat.shape[0], 2)
        for idx, vec in enumerate(sample_data):
            # This looks gross, but is just comparing the row vectors one at a time
            self.assertEqual(vec, stacked_mat[idx].toarray().tolist()[0])

    def test_vectorization_stack_matrices_fails_when_disallowed(self):
        """Test that we explode when stacking matrices if we explicitly disallow matrices."""
        sample_data = [[1, 2, 3, 4], [[5, 6, 7, 8], [9, 10, 11, 12]]]
        vec_preds = [
            dm.VectorizationPrediction(sparse.csr_matrix(vec)) for vec in sample_data
        ]
        with self.assertRaises(ValueError):
            dm.SparseMatrix.stack_raw_tensor(vec_preds, allow_matrices=False)

    def test_vectorization_stack_matrices_succeeds_when_allowed(self):
        """Test that we explode when stacking matrices if we explicitly allow matrices."""
        sample_data = [[1, 2, 3, 4], [[5, 6, 7, 8], [9, 10, 11, 12]]]
        # Expected flat result after nesting is considered and done
        expected_data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        vec_preds = [
            dm.VectorizationPrediction(sparse.csr_matrix(vec)) for vec in sample_data
        ]
        stacked_mat = dm.SparseMatrix.stack_raw_tensor(vec_preds, allow_matrices=True)
        self.assertIsInstance(stacked_mat, sparse.csr_matrix)
        # Stacking a (1 x n) + (2 x n) should give us a (3 x n)
        self.assertEqual(stacked_mat.shape[0], 3)
        for idx, vec in enumerate(expected_data):
            self.assertEqual(vec, stacked_mat[idx].toarray().tolist()[0])

    def test_invalid_dtype_str(self):
        with self.assertRaises(TypeError):
            dm.SparseMatrix(
                data=[1, 4],
                indices=[0, 1],
                indptr=[0, 1, 2],
                rows=2,
                cols=2,
                dtype="float642",
            )

    def test_invalid_dtype_type(self):
        with self.assertRaises(TypeError):
            dm.SparseMatrix(
                data=[1, 4], indices=[0, 1], indptr=[0, 1, 2], rows=2, cols=2, dtype=int
            )

    def test_default_matrix_dtype(self):
        self.assertEqual(
            np.dtype(
                dm.SparseMatrix(
                    data=[1, 4], indices=[0, 1], indptr=[0, 1, 2], rows=2, cols=2
                ).dtype
            ),
            np.dtype("float32"),
        )

    def test_int32_matrix(self):
        self.assertEqual(
            np.dtype(
                dm.SparseMatrix(
                    data=[1, 4],
                    indices=[0, 1],
                    indptr=[0, 1, 2],
                    rows=2,
                    cols=2,
                    dtype="int32",
                ).dtype
            ),
            np.dtype("int32"),
        )

    def test_float64_matrix(self):
        self.assertEqual(
            np.dtype(
                dm.SparseMatrix(
                    data=[1, 4],
                    indices=[0, 1],
                    indptr=[0, 1, 2],
                    rows=2,
                    cols=2,
                    dtype="float64",
                ).dtype
            ),
            np.dtype("float64"),
        )

    def test_int64_matrix(self):
        self.assertEqual(
            np.dtype(
                dm.SparseMatrix(
                    data=[1, 4],
                    indices=[0, 1],
                    indptr=[0, 1, 2],
                    rows=2,
                    cols=2,
                    dtype="int64",
                ).dtype
            ),
            np.dtype("int64"),
        )

    def test_from_numpy_int32(self):
        self.assertEqual(
            np.dtype(
                dm.SparseMatrix.from_scipy_csr_matrix(
                    sparse.csr_matrix(
                        ([1, 4], [0, 1], [0, 1, 2]), dtype=np.dtype("int32")
                    )
                ).dtype
            ),
            np.dtype("int32"),
        )

    def test_to_numpy_int32(self):
        self.assertEqual(
            dm.SparseMatrix.to_scipy_csr_matrix(
                dm.SparseMatrix(
                    data=[1, 4],
                    indices=[0, 1],
                    indptr=[0, 1, 2],
                    rows=2,
                    cols=2,
                    dtype="int32",
                )
            ).dtype,
            np.dtype("int32"),
        )

    def test_to_dict_int32(self):
        self.assertEqual(
            dm.SparseMatrix.to_dict(
                dm.SparseMatrix(
                    data=[1, 4],
                    indices=[0, 1],
                    indptr=[0, 1, 2],
                    rows=2,
                    cols=2,
                    dtype="int32",
                )
            )["dtype"],
            "int32",
        )
