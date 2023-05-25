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
"""Data structures for dense vector / matrix representations.
"""
# Standard
from typing import Dict, List

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import matrix

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class Embedding(DataObjectBase):
    """An Embedding object, acts as a lookup table, usually mapping
    the vocabulary to vectors. Note that the `data` property on Embedding/EmbeddingPrediction is a
    Numpy array for numerical convenience - DenseMatrix objects are built only temporarily for
    conversions in order to tie the data model to the proto spec.
    """

    data: Annotated[matrix.DenseMatrix, FieldNumber(1)]
    vocab_to_idx: Annotated[Dict[str, int], FieldNumber(2)]
    producer_id: Annotated[ProducerId, FieldNumber(3)]
    pad_word: Annotated[str, FieldNumber(4)]
    unk_word: Annotated[str, FieldNumber(5)]

    def __init__(
        self, data, vocab_to_idx, pad_word="<pad>", unk_word="<unk>", producer_id=None
    ):
        """Initialize a new instance of this class.

        Args:
            data: np.ndarray
                2D numpy array representing the Embedding.
            vocab_to_idx: dict
                Mapping from strings to row indices.
            pad_word: string
                Pad word that was used for training.
            unk_word: string
                Unknown word that was used for training.
            producer_id:  ProducerId or None
                The block that produced this embedding.
        """
        # Require data to be a numpy array - will only be converted to DenseMatrix when converting
        if isinstance(data, matrix.DenseMatrix):
            error(
                "<NLP56497475E>",
                TypeError(
                    "Although the proto spec uses DenseMatrix objects for serialization, [data] "
                    + "is represented as a numpy array on this class for convenience. Pass this "
                    + "initializer the result of the to numpy conversion method on this "
                    + "DenseMatrix instance."
                ),
            )
        error.type_check("<NLP70983280E>", np.ndarray, data=data)
        error.type_check(
            "<NLP97572539E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        if not isinstance(vocab_to_idx, dict):
            error("<NLP26735495E>", TypeError("`vocab` needs to be dictionary"))
        # Require vocab dict to be a <string, int> pair.
        if not all(
            isinstance(k, str) and isinstance(v, int) for k, v in vocab_to_idx.items()
        ):
            error(
                "<NLP09635598W>",
                TypeError("`vocab` key/value pair is not a <string, int> pair"),
            )
        super().__init__()
        if data.dtype != np.float32:
            log.warning(
                "<NLP14004432W>",
                "Data type {} will be converted to float 32.".format(data.dtype),
            )
            data = data.astype(np.float32)
        self.data = data
        self.vocab_to_idx = vocab_to_idx
        self.pad_word = pad_word
        self.unk_word = unk_word
        self.producer_id = producer_id

    def fill_proto(self, proto):
        """Override for filling proto on embedding - this is necessary because we need
        to serialize a numpy array. This is accomplished by first converting it to a DenseMatrix
        data model object and then converting that to protobuf.

        Args:
            proto: embedding_types_pb2.Embedding
                The embedding prediction protobuf class object.
        Returns:
            protobuf
                A DataBase object
        """
        # Convert the data numpy array to a DenseMatrix so that we can to_proto it
        proto_data = matrix.DenseMatrix.from_numpy_array(self.data).to_proto()
        proto.data.CopyFrom(proto_data)
        proto.vocab_to_idx.update(self.vocab_to_idx)
        proto.pad_word = self.pad_word
        proto.unk_word = self.unk_word
        if self.producer_id is not None:
            proto_producer = self.producer_id.to_proto()
            proto.producer_id.CopyFrom(proto_producer)
        return proto

    @classmethod
    def from_proto(cls, proto):
        """Override for creating an instance of this class from an Embedding protobuf
        class instance. Note that DenseMatrix is loaded onto the data property as a numpy array
        for convenience.

        Args:
            proto: embedding_types_pb2.Embedding
                The embedding protobuf class object.
        Returns:
            Embedding
                An instance of this class.
        """
        # Producer is optional & a data model class - if we have it, we can from_proto it
        prod = ProducerId.from_proto(proto.producer_id) if proto.producer_id else None
        # Pull a dense matrix out of the proto, then convert that to a Numpy array
        data = matrix.DenseMatrix.from_proto(proto.data).to_numpy_array()
        vocab_to_idx = dict(proto.vocab_to_idx)
        kwargs = {"data": data, "vocab_to_idx": vocab_to_idx, "producer_id": prod}
        if proto.pad_word:
            kwargs["pad_word"] = proto.pad_word
        if proto.unk_word:
            kwargs["unk_word"] = proto.unk_word
        return cls(**kwargs)

    def to_dict(self):
        """Override for dictionary conversion for this object, which is used by .to_json() - we do
        this so that we can stringify the data numpy array. Note that we only need to handle this
        for to_json! We get from_json for free because it leverages the other overrides for proto
        conversions.

        Returns:
            dict
                Dictionary representation of an instance of this class.
        """
        return {
            "data": matrix.DenseMatrix.from_numpy_array(self.data).to_dict(),
            "vocab_to_idx": self.vocab_to_idx,
            "pad_word": self.pad_word,
            "unk_word": self.unk_word,
            "producer_id": self.producer_id.to_dict() if self.producer_id else None,
        }

    def get_padded_token_indices(self, tokens, max_len):
        """Function to add a pad word after each sentence
            if it is smaller then max_len else truncate
        Args:
            tokens: list(str)
                input list of tokenized documents
            max_len:
                Max length of the tokens

        Returns:
            np.array: Numpy array of token indices
        """

        # Initialize the token_indices with pad word's index
        token_indices = np.full((max_len,), self.vocab_to_idx[self.pad_word])
        unk_word_index = self.vocab_to_idx[self.unk_word]

        for idx in range(min(max_len, len(tokens))):
            # If vocab_to_idx contains the token, get token's value else use unk_word_index
            token_indices[idx] = self.vocab_to_idx.get(tokens[idx], unk_word_index)

        return token_indices

    def get_ordered_vocab(self):
        """Retrieve the vocabulary of the model in order, such the the index of each token
        corresponds to its row vector in the data matrix.

        Returns:
            list(str)
                List of ordered tokens.
        """
        sorted_kv_pairs = sorted(self.vocab_to_idx.items(), key=lambda x: x[1])
        return [key for (key, _) in sorted_kv_pairs]


@dataobject(package="caikit_data_model.nlp")
class EmbeddingPrediction(DataObjectBase):
    data: Annotated[matrix.DenseMatrix, FieldNumber(1)]
    offsets: Annotated[List[np.uint32], FieldNumber(2)]
    producer_id: Annotated[ProducerId, FieldNumber(3)]

    """An EmbeddingPrediction object, which describes the result of running an Embedding model
    over an input text. Note that the `data` property on Embedding/EmbeddingPrediction is a
    Numpy array for numerical convenience - DenseMatrix objects are built only temporarily for
    conversions in order to tie the data model to the proto spec.
    """

    def __init__(self, data, offsets=None, producer_id=None):
        """Initialize a new instance of this class.

        Args:
            data: np.ndarray
                (n x m) numpy array representing the Embedding Prediction.
            offsets: tuple(int) | list(int) | np.ndarray(int) | None
                Row offset indices indicating which vectors belong to which input objects, e.g.,
                documents. Offsets here refer to the row indices at which new documents are formed,
                where the initial entry (0) is optional. For example, providing (2,) is equivalent
                to providing (0, 2,) and implies that the first 2 rows of the EmbeddingPrediction
                data were generated by considering one input source, while the final (n-2) rows
                were generated considering a second input source.
            producer_id:  ProducerId or None
                The block that produced this embedding prediction.
        """
        # Require data to be a numpy array - will only be converted to DenseMatrix when converting
        if isinstance(data, matrix.DenseMatrix):
            error(
                "<NLP51000151E>",
                TypeError(
                    "Although the proto spec uses DenseMatrix objects for serialization, [data] "
                    + "is represented as a numpy array on this class for convenience. Pass this "
                    + "initializer the result of the to numpy conversion method on this "
                    + "DenseMatrix instance."
                ),
            )
        error.type_check("<NLP25119393E>", np.ndarray, data=data)
        error.type_check(
            "<NLP51313700E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        error.type_check(
            "<NLP51391744E>", tuple, list, np.ndarray, allow_none=True, offsets=offsets
        )
        super().__init__()
        offsets = EmbeddingPrediction._validate_and_standardize_offset_format(
            data, offsets
        )
        if data.dtype != np.float32:
            log.warning(
                "<NLP14004431W>",
                "Data type {} will be converted to float 32.".format(data.dtype),
            )
            data = data.astype(np.float32)
        self.data = data
        self.offsets = offsets
        self.producer_id = producer_id

    @staticmethod
    def _validate_and_standardize_offset_format(data, offsets):
        """Validate and standardize offsets such that:
            - If the offsets represent one input sources, the instance offsets are set to None
            - If they represent multiple input sources, the instance offsets are sorted in
              ascending order, starting with 0 (which may or may not be provided).
        Args:
            offsets: tuple(int) | list(int) | np.ndarray(int) | None
                Row offset indices indicating which vectors belong to which input objects, e.g.,
                documents. Offsets here refer to the row indices at which new documents are formed,
                where the initial entry (0) is optional.
        """
        # Single document cases - None was provided, we were given an empty list/tuple, or we
        # were just given a single element list or tuple containing only 0, which covers everything
        if not offsets or (len(offsets) == 1 and not offsets[0]):
            return None
        num_rows = data.shape[0]
        # Sort offsets in ascending order
        sorted_offsets = sorted(offsets)
        # Ensure that we don't have any duplicate values or things out of range
        error.value_check(
            "<NLP74114171E>",
            len(sorted_offsets) == len(set(sorted_offsets)),
            "Provided offsets must contain unique row index values into the data matrix!",
        )
        error.type_check_all("<NLP74115571E>", int, sorted_offsets=sorted_offsets)
        error.value_check(
            "<NLP74114471E>",
            all(0 <= val < num_rows for val in sorted_offsets),
            "Starting row offsets must be in the range [0, n) to contain at least one row!",
        )
        # Add 0 if we don't have it already
        if sorted_offsets[0]:
            sorted_offsets.insert(0, 0)
        return sorted_offsets

    def fill_proto(self, proto):
        """Override for filling proto on embedding predictions - this is necessary because we need
        to serialize a numpy array. This is accomplished by first converting it to a DenseMatrix
        data model object and then converting that to protobuf.

        Args:
            proto: embedding_types_pb2.EmbeddingPrediction
                The embedding prediction protobuf class object.
        Returns:
            protobuf
                A DataBase object
        """
        # Convert the data numpy array to a DenseMatrix so that we can to_proto it
        proto_data = matrix.DenseMatrix.from_numpy_array(self.data).to_proto()
        proto.data.CopyFrom(proto_data)
        if self.producer_id is not None:
            proto_producer = self.producer_id.to_proto()
            proto.producer_id.CopyFrom(proto_producer)
        return proto

    @classmethod
    def from_proto(cls, proto):
        """Override for creating an instance of this class from an EmbeddingPrediction protobuf
        class instance. Note that DenseMatrix is loaded onto the data property as a numpy array
        for convenience.

        Args:
            proto: embedding_types_pb2.EmbeddingPrediction
                The embedding prediction protobuf class object.
        Returns:
            EmbeddingPrediction
                An instance of this class.
        """
        # Pull a dense matrix out of the proto, then convert that to a Numpy array
        data = matrix.DenseMatrix.from_proto(proto.data).to_numpy_array()
        offsets = tuple(proto.offsets) if proto.offsets else None
        # Producer is optional & a data model class - if we have it, we can from_proto it
        prod = ProducerId.from_proto(proto.producer_id) if proto.producer_id else None
        return cls(data, offsets, prod)

    def to_dict(self):
        """Override for dictionary conversion for this object, which is used by .to_json() - we do
        this so that we can stringify the data numpy array. Note that we only need to handle this
        for to_json! We get from_json for free because it leverages the other overrides for proto
        conversions.

        Returns:
            dict
                Dictionary representation of an instance of this class.
        """
        return {
            "data": matrix.DenseMatrix.from_numpy_array(self.data).to_dict(),
            "offsets": self.offsets,
            "producer_id": self.producer_id.to_dict() if self.producer_id else None,
        }

    def disassemble(self):
        """Break the current embedding object into a list of Embedding objects, where objects
        are split apart by their starting row offsets. For example, given an EmbeddingPrediction
        with 10 rows, and starting offsets (0, 5), produce two EmbeddingPrediction objects, the
        first of which contains the data from rows [0, 4], and the second containing data from rows
        [5, 10). This method should be used to unstack outputs from run_batch() extracted from
        Embedding modules, where each input source contains multiple vectors, e.g., sentence
        embeddings.

        Returns:
            list(EmbeddingPrediction)
                EmbeddingPrediction objects corresponding to sliced out input sources.
        """
        error.value_check(
            "<NLP74113171E>",
            self.offsets is not None,
            "EmbeddingPrediction has no offsets; there is nothing to disassemble!",
        )
        # Add the number of rows to the offset list so that we can slice adjacent entries
        full_offsets = self.offsets + [self.data.shape[0]]
        extract_submat = lambda idx: self.data[
            full_offsets[idx] : full_offsets[idx + 1], :
        ]
        # Extract submatrices corresponding to adjacent offsets and build EmbeddingPredictions
        # corresponding to individual input sources / documents.
        return [
            EmbeddingPrediction(extract_submat(idx))
            for idx in range(len(full_offsets) - 1)
        ]
