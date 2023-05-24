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
"""Data structures for interacting with dense/sparse matrices. At the time of implementation, we
use dense matrices for embeddings and sparse matrices for vectorizations. However, since these data
structures may be more commonly useful one day, we move them out into a separate class file here.
"""

# Standard
from inspect import signature
from typing import List

# Third Party
from scipy import sparse
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


def _stack_vectors(mat_iter, stack_func, allow_matrices):
    """Common function for stacking EmbeddingPredictions/VectorizationPredictions into their
    corresponding encapsulated raw matrix types.

    Args:
        mat_iter: list | tuple | ....core.data_model.data_stream.DataStream
            An iterable of VectorizationPredictions or EmbeddingPredictions whose data matrices
            we would like to stack into a single matrix for later processing.
        stack_func: function
            Function accepting a single parameter (list of raw matrices) which defines the
            stacking behavior for the type being considered.
        allow_matrices: bool
            Whether or not we should explode if any of the prediction objects contained in the
            provided iterable contain multiple rows.

    Returns:
        np.ndarray | scipy.sparse.csr_matrix
            Output of stacking func, type depending on class leveraging this function.
    """
    # If we don't allow matrices and we find somthing with multiple rows, explode!
    if not allow_matrices and not all(mat.data.shape[0] == 1 for mat in mat_iter):
        error(
            "<NLP51491475E>",
            ValueError(
                "Encountered disallowed data matrix while executing vector stacking"
            ),
        )
    # Otherwise pull out the list of things to be stacked and execute the provided func
    data_arrs = [pred.data for pred in mat_iter]
    return stack_func(data_arrs)


@dataobject(package="watson_core_data_model.nlp")
class DenseMatrix(DataObjectBase):
    data: Annotated[List[float], FieldNumber(1)]
    rows: Annotated[np.uint32, FieldNumber(2)]
    cols: Annotated[np.uint32, FieldNumber(3)]
    dtype: Annotated[str, FieldNumber(4)]

    """A dense matrix, used for representing embedding data in a serialized format. Note that
    the `data` property on Embedding/EmbeddingPrediction will be a Numpy array since this is much
    more convenient to work with numerically. This type is primarily used for serialization and
    deserialization (proto & json).
    """

    def __init__(self, data, rows, cols, dtype="float32"):
        """Construct a new DenseMatrix.

        Args:
            data: list | tuple
                flattened numpy matrix.
            rows: int
                Number of rows in the data matrix (>=0).
            cols: int
                Number of cols in the data matrix (>=0).
            dtype: str
                The numpy data type, represented as string. Default is float32.

        """
        error.type_check("<NLP76181661E>", list, tuple, data=data)
        error.type_check("<NLP61830610E>", int, rows=rows)
        error.type_check("<NLP50082136E>", int, cols=cols)
        error.type_check("<NLP74269231E>", str, dtype=dtype)
        error.value_check(
            "<NLP08696007E>",
            rows * cols == len(data),
            "Product of rows and columns must equal the length of flattened data.",
        )
        error.value_check(
            "<NLP44616450E>",
            rows >= 0 and cols >= 0,
            "DenseMatrix requires nonnegative row/col dimensions.",
        )
        # If our dtype is explicitly set to empty, grab the default value and set it directly.
        # This can occur in some special cases, like when we load a CNN classifier containing
        # an Embedding data model object from a binary buffer, because we may be calling from_proto
        # on objects where dtype wasn't yet added to the data model, and protobuf gives strings
        # the default value of empty.
        if not len(dtype):
            dtype = signature(self.__init__).parameters["dtype"].default
        # check that the dtype is a string representing a numpy data type
        try:
            np.dtype(dtype)
        except TypeError as e:
            error(
                "<NLP13830871E>",
                TypeError("arg [dtype] has invalid value: " + str(dtype)),
            )
        super().__init__()
        self.data = data
        self.rows = rows
        self.cols = cols
        self.dtype = dtype

    @property
    def num_texts(self):
        """Property alias for rows.

        Returns:
            int
                Number of rows in the matrix.
        """
        return self.rows

    @num_texts.setter
    def num_texts(self, new_value):
        """Setter for num_texts to ensure values are at least aligned with rows.

        Args:
            new_value: int
                new value for rows. No type checking is done here since you could just set rows
                directly, but be a responsible coder!
        """
        self.rows = new_value

    @property
    def num_dims(self):
        """Property alias for cols.

        Returns:
            int
                Number of cols in the matrix.
        """
        return self.cols

    @num_dims.setter
    def num_dims(self, new_value):
        """Setter for num_dums to ensure values are at least aligned with cols.

        Args:
            new_value: int
                new value for cols. No type checking is done here since you could just set cols
                directly, but be a responsible coder!
        """
        self.cols = new_value

    def to_numpy_array(self):
        """Returns the numpy array which this object represents.

        Returns:
            numpy.ndarray
                numpy array with two axes.
        """
        return np.array(self.data, dtype=np.dtype(self.dtype)).reshape(
            self.rows, self.cols
        )

    @classmethod
    def from_numpy_array(cls, data):
        """Initialize an instance of this class from a 2D numpy array.

        Args:
            data: numpy.ndarray
                numpy array with two axes.
        Returns:
            DenseMatrix
                JSON / proto serializable representation of the data matrix.
        """
        error.type_check("<NLP94565248E>", np.ndarray, data=data)
        error.value_check(
            "<NLP16215129E>",
            len(data.shape) == 2,
            "DenseMatrix construction requires a numpy array with 2 axes.",
        )
        rows, cols = data.shape
        dtype = str(data.dtype)
        # Data should be immutable given that rows/cols are fixed & must be json serializable
        # to play nicely with the data model, so we can't use numpy arrays out of the box. We
        # use list as an intermediate container since it converts numpy types to python's native.
        data = tuple(list(data.flatten()))
        return cls(data, rows, cols, dtype)

    def to_dict(self):
        """Override for json serialization, since we use numpy types to represent matrices, which are
        not JSON serializable. We convert to float since this is the more generic data type that we can
        use in the protobuf. By default, python's float type is np.float64, so there is no type
        mapping in that case. If the original numpy type is different, we convert back.
        """
        return {
            "data": [float(datum) for datum in self.data],
            "rows": self.rows,
            "cols": self.cols,
            "dtype": self.dtype,
        }

    @staticmethod
    def stack_raw_tensor(mat_list, allow_matrices=True):
        """Given a list, tuple, or DataStream of EmbeddingPredictions, consolidate the data
        attributes into a single numpy matrix. As a precondition, this method assumes mat_iter
        has been properly inspected to avoid excessive type checks, as this will generally be
        leveraged internally by blocks that have their own homogeneity checks on iterable args.

        Args:
            mat_list: list | tuple | ....core.data_model.data_stream.DataStream
                An iterable of EmbeddingPredictions whose data matrices we would like to stack
                into a single numpy matrix for later processing.
            allow_matrices: bool
                Whether or not we should explode if any of the prediction objects contained in the
                provided iterable contain multiple rows.

        Returns:
            numpy.ndarray
                numpy matrix with two axes for later use.
        """

        def stack_func(data_arrs):
            return np.concatenate(data_arrs, axis=0)

        return _stack_vectors(mat_list, stack_func, allow_matrices)


@dataobject(package="watson_core_data_model.nlp")
class SparseMatrix(DataObjectBase):
    data: Annotated[List[float], FieldNumber(1)]
    indices: Annotated[List[np.uint32], FieldNumber(2)]
    indptr: Annotated[List[np.uint32], FieldNumber(3)]
    rows: Annotated[np.uint32, FieldNumber(4)]
    cols: Annotated[np.uint32, FieldNumber(5)]
    dtype: Annotated[str, FieldNumber(6)]

    """A sparse matrix, used for representing vectorized data in a serialized format. Note that
    the `data` property on Vectorization/VectorizationPrediction will be a Scipy array since this
    is much more convenient to work with numerically. This type is primarily used for serialization
    and deserialization (proto & json).
    """

    def __init__(self, data, indices, indptr, rows, cols, dtype="float32"):
        """Construct a new SparseMatrix using properties affiliated with CSR matrix format.

        Args:
            data: list | tuple
                flattened nonzero data entries.
            indices: list | tuple
                Column indices [aligned with nz data entries].
            indptr: list | tuple
                List of [data] indices that start new rows.
            rows: int
                Number of rows in the data matrix (>=0).
            cols: int
                Number of cols in the data matrix (>=0).
            dtype: str
                The numpy data type, represented as string. Default is float32.
        """
        error.type_check("<NLP79565166E>", list, tuple, data=data)
        error.type_check("<NLP88183167E>", list, tuple, indices=indices)
        error.type_check("<NLP88034506E>", list, tuple, indptr=indptr)
        error.type_check("<NLP09659332E>", int, rows=rows)
        error.type_check("<NLP96861267E>", int, cols=cols)
        error.type_check("<NLP50113170E>", str, dtype=dtype)
        error.value_check(
            "<NLP34544923E>",
            rows >= 0 and cols >= 0,
            "SparseMatrix requires nonnegative row/col dimensions.",
        )
        error.value_check(
            "<NLP08633337E>",
            len(data) <= rows * cols,
            "Number of [data] entries may not exceed product of rows and columns.",
        )
        error.value_check(
            "<NLP27934809E>",
            len(data) == len(indices),
            "args [data] and [indices] must be the same length for CSR format.",
        )
        error.value_check(
            "<NLP39739722E>",
            max(indptr) <= len(data),
            "arg [indptr] may not have index values exceeding the length of [data].",
        )
        error.value_check(
            "<NLP68768503E>",
            sorted(indptr) == indptr,
            "arg [indptr] must be sorted in ascending order in CSR format.",
        )
        error.value_check(
            "<NLP17168385E>",
            len(set(indptr)) == len(indptr),
            "arg [indptr] may not contain duplicate values in CSR format.",
        )
        # If our dtype is explicitly set to empty, grab the default value and set it directly.
        # This can occur in some special cases, like when we load a CNN classifier containing
        # an Embedding data model object from a binary buffer, because we may be calling from_proto
        # on objects where dtype wasn't yet added to the data model, and protobuf gives strings
        # the default value of empty.
        if not len(dtype):
            dtype = signature(self.__init__).parameters["dtype"].default
        # check that the dtype is a string representing a numpy data type
        try:
            np.dtype(dtype)
        except TypeError as e:
            error(
                "<NLP50343527E>",
                TypeError(f"arg [dtype] has invalid value: {str(dtype)}"),
            )
        super().__init__()
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.rows = rows
        self.cols = cols
        self.dtype = dtype

    @property
    def num_texts(self):
        """Property alias for rows.

        Returns:
            int
                Number of rows in the matrix.
        """
        return self.rows

    @num_texts.setter
    def num_texts(self, new_value):
        """Setter for num_texts to ensure values are at least aligned with rows.

        Args:
            new_value: int
                new value for rows. No type checking is done here since you could just set rows
                directly, but be a responsible coder!
        """
        self.rows = new_value

    @property
    def num_dims(self):
        """Property alias for cols.

        Returns:
            int
                Number of cols in the matrix.
        """
        return self.cols

    @num_dims.setter
    def num_dims(self, new_value):
        """Setter for num_dums to ensure values are at least aligned with cols.

        Args:
            new_value: int
                new value for cols. No type checking is done here since you could just set cols
                directly, but be a responsible coder!
        """
        self.cols = new_value

    def to_scipy_csr_matrix(self):
        """Returns the numpy array which this object represents.

        Returns:
            scipy.sparse.csr_matrix
                Sparse CSR matrix that this object represents.
        """
        return sparse.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.rows, self.cols),
            dtype=np.dtype(self.dtype),
        )

    @classmethod
    def from_scipy_csr_matrix(cls, data_mat_csr):
        """Initialize an instance of this class from a scipy sparse csr matrix.

        Args:
            data_mat_csr: scipy.sparse.csr_matrix
                sparse csr matrix that we will represent in an instance of this class.
        Returns:
             SparseMatrix
                JSON / proto serializable representation of the data matrix.
        """
        error.type_check("<NLP98572907E>", sparse.csr_matrix, data_mat_csr=data_mat_csr)
        rows, cols = data_mat_csr.shape
        dtype = str(data_mat_csr.dtype)
        # By default, data/indices/indptr are stored as 1D numpy arrays. tolist() will convert to
        # built-in Python types, which is cleaner to handle for this class, since the result will
        # be serializable. Numpy arrays generally need custom overrides.
        tup_data = data_mat_csr.data.tolist()
        tup_indices = data_mat_csr.indices.tolist()
        tup_indptr = data_mat_csr.indptr.tolist()
        return cls(tup_data, tup_indices, tup_indptr, rows, cols, dtype)

    def to_dict(self):
        """Override for json serialization, since we use numpy types to represent matrices, which are
        not JSON serializable (by default, python's float type is np.float64, so there is no type
        mapping in that case).
        """
        return {
            "data": [float(datum) for datum in self.data],
            "indices": self.indices,
            "indptr": self.indptr,
            "rows": self.rows,
            "cols": self.cols,
            "dtype": self.dtype,
        }

    @staticmethod
    def stack_raw_tensor(mat_iter, allow_matrices=True):
        """Given a list, tuple, or DataStream of VectorizationPredictions, consolidate the data
        attributes into a single scipy matrix. As a precondition, this method assumes mat_iter
        has been properly inspected to avoid excessive type checks, as this will generally be
        leveraged internally by blocks that have their own homogeneity checks on iterable args.

        Args:
            mat_iter: list | tuple | ....core.data_model.data_stream.DataStream
                An iterable of VectorizationPredictions whose data matrices we would like to stack
                into a single sparse matrix for later processing.
            allow_matrices: bool
                Whether or not we should explode if any of the prediction objects contained in the
                provided iterable contain multiple rows.

        Returns:
            scipy.sparse.csr_matrix
                Scipy CSR matrix for later use.
        """
        return _stack_vectors(mat_iter, sparse.vstack, allow_matrices)
