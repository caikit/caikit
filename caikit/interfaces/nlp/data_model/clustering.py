# *************************************************************** #
# (C) Copyright IBM Corporation 2021.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *************************************************************** #
"""Data structures for text clustering.
"""
# Standard
from typing import List

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit import error_handler
from ...common.data_model import ProducerId
from . import matrix

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="watson_core_data_model.nlp")
class ClusteringPrediction(DataObjectBase):
    """A clustering analysis generated from multiple vectors."""

    cluster_ids: Annotated[List[np.int32], FieldNumber(1)]
    costs: Annotated[matrix.DenseMatrix, FieldNumber(2)]
    producer_id: Annotated[ProducerId, FieldNumber(3)]

    """The result of a clustering analysis."""

    def __init__(self, cluster_ids, costs, producer_id=None):
        """Construct a new clustering prediction object.
        Args:
            cluster_ids:  np.ndarray
                The cluster id to which each sample is assigned.
                Represented as a 1d array of int values of size
                n_samples.
            costs:  np.ndarray
                The distance between each sample and each centroid.
                Represented as a 2d array of float values of the
                shape [n_samples x n_clusters].
            producer_id:  ProducerId or None
                The block that produced this clustering prediction.
        """

        # type checks
        if isinstance(cluster_ids, (list, tuple)):
            cluster_ids = np.array(cluster_ids)
        error.type_check("<NLP48364205E>", np.ndarray, cluster_ids=cluster_ids)
        if isinstance(costs, matrix.DenseMatrix):
            error(
                "<NLP13896110E>",
                TypeError(
                    "Although the proto spec uses DenseMatrix objects for serialization, [costs] "
                    + "is represented as a numpy array on this class for convenience. Pass this "
                    + "initializer the result of the to numpy conversion method on this "
                    + "DenseMatrix instance."
                ),
            )
        error.type_check("<NLP13817132E>", np.ndarray, costs=costs)
        error.type_check(
            "<NLP91718764E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        # dimension checks
        error.value_check(
            "<NLP60160233E>",
            np.ndim(cluster_ids) == 1,
            "The cluster ids array must be 1d",
        )
        error.value_check(
            "<NLP85474600E>", np.ndim(costs) == 2, "The costs array must be 2d"
        )

        # boundary checks for cluster_ids
        error.value_check(
            "<NLP04150746E>",
            cluster_ids.min() >= 0,
            "`cluster_id` of `{}` is invalid",
            cluster_ids.min(),
        )
        error.value_check(
            "<NLP45846813E>",
            cluster_ids.max() <= costs.shape[1],
            "`cluster_id` of `{}` is invalid (too high)",
            cluster_ids.max(),
        )

        # boundary check for costs
        error.value_check(
            "<NLP69180101E>", costs.min() >= 0, "`cost` of `{}` is invalid", costs.min()
        )

        # size matching check between cluster_ids and costs
        error.value_check(
            "<NLP10629772E>",
            len(cluster_ids) == costs.shape[0],
            "Mismatch between number of samples in cluster ids `{}` and in costs `{}`",
            len(cluster_ids),
            costs.shape[0],
        )

        super().__init__()
        self.cluster_ids = cluster_ids
        self.costs = costs
        self.producer_id = producer_id

    def fill_proto(self, proto):
        """Override for filling proto on clustering predictions - this is necessary because we
        need to serialize two numpy arrays - cluster_ids and costs. This is accomplished by:
        (a) converting cluster ids to a list
        (b) converting costs to a DenseMatrix data model object and then converting that to protobuf.

        Args:
            proto: clustering_types_pb2.ClusteringPrediction
                The clustering prediction protobuf class object.
        Returns:
            protobuf
                A DataBase object
        """

        # Convert the cluster ids 1d numpy array to a list
        proto.cluster_ids.extend(self.cluster_ids.tolist())

        # Convert the costs 2d numpy array to a DenseMatrix so that we can to_proto it
        proto.costs.CopyFrom(matrix.DenseMatrix.from_numpy_array(self.costs).to_proto())

        # add producer id if we have it
        if self.producer_id is not None:
            proto.producer_id.CopyFrom(self.producer_id.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto):
        """Override for creating an instance of this class from a ClusteringPrediction protobuf
        class instance. Note that the a DenseMatrix is loaded onto the costs property as a numpy array
        for convenience. Similarly, a list is loaded onto the cluster_ids property as a a numpy array.

        Args:
            proto: clustering_types_pb2.ClusteringPrediction
                The clustering prediction protobuf class object.
        Returns:
            ClusteringPrediction
                An instance of this class.
        """
        # Producer is optional & a data model class - if we have it, we can from_proto it
        prod = ProducerId.from_proto(proto.producer_id) if proto.producer_id else None

        # Pull a dense matrix out of the proto, then convert that to a numpy array
        costs = matrix.DenseMatrix.from_proto(proto.costs).to_numpy_array()

        # convert the cluster ids list to numpy array
        cluster_ids = np.array(proto.cluster_ids)

        return cls(cluster_ids, costs, prod)

    def to_dict(self):
        """Override for dictionary conversion for this object, which is used by .to_json() - we do
        this so that we can stringify the numpy arrays of cluster_ids and costs. Note that we only
        need to handle this for to_json! We get from_json for free because it leverages the other
        overrides for proto conversions.

        Returns:
            dict
                Dictionary representation of an instance of this class.
        """
        return {
            "cluster_ids": self.cluster_ids.tolist(),
            "costs": matrix.DenseMatrix.from_numpy_array(self.costs).to_dict(),
            "producer_id": self.producer_id.to_dict() if self.producer_id else None,
        }
