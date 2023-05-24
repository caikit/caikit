# *****************************************************************#
# (C) Copyright IBM Corporation 2021.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Third Party
import numpy as np
import utils

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase

np.random.seed(1024)


class TestClusteringPrediction(TestCaseBase):
    def setUp(self):
        costs = np.random.rand(6, 3)
        cluster_ids = np.argmin(costs, axis=1)
        self.clustering_prediction = dm.ClusteringPrediction(
            cluster_ids=cluster_ids,
            costs=costs,
            producer_id=dm.ProducerId("Test", "1.2.3"),
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.clustering_prediction))

    def test_from_proto_and_back(self):
        new = dm.ClusteringPrediction.from_proto(self.clustering_prediction.to_proto())
        self.assertTrue(
            np.allclose(new.cluster_ids, self.clustering_prediction.cluster_ids)
        )
        self.assertTrue(np.allclose(new.costs, self.clustering_prediction.costs))

    def test_from_json_and_back(self):
        new = dm.ClusteringPrediction.from_json(self.clustering_prediction.to_json())
        self.assertTrue(
            np.allclose(new.cluster_ids, self.clustering_prediction.cluster_ids)
        )
        self.assertTrue(np.allclose(new.costs, self.clustering_prediction.costs))

    def test_list_cluster_ids(self):
        self.assertTrue(
            utils.validate_fields(
                dm.ClusteringPrediction(
                    cluster_ids=list(self.clustering_prediction.cluster_ids),
                    costs=self.clustering_prediction.costs,
                    producer_id=dm.ProducerId("Test", "1.2.3"),
                )
            )
        )

    def test_tuple_cluster_ids(self):
        self.assertTrue(
            utils.validate_fields(
                dm.ClusteringPrediction(
                    cluster_ids=tuple(self.clustering_prediction.cluster_ids),
                    costs=self.clustering_prediction.costs,
                    producer_id=dm.ProducerId("Test", "1.2.3"),
                )
            )
        )

    def test_invalid_cost_negative(self):
        new_costs = self.clustering_prediction.costs.copy()
        new_costs[3, 2] = -0.1
        with self.assertRaises(ValueError):
            dm.ClusteringPrediction(
                self.clustering_prediction.cluster_ids,
                new_costs,
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_invalid_cluster_id_negative(self):
        new_cluster_ids = self.clustering_prediction.cluster_ids.copy()
        new_cluster_ids[2] = -1
        with self.assertRaises(ValueError):
            dm.ClusteringPrediction(
                new_cluster_ids,
                self.clustering_prediction.costs,
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_invalid_cluster_id_too_high(self):
        new_cluster_ids = self.clustering_prediction.cluster_ids.copy()
        new_cluster_ids[5] = self.clustering_prediction.costs.shape[0] + 1
        with self.assertRaises(ValueError):
            dm.ClusteringPrediction(
                new_cluster_ids,
                self.clustering_prediction.costs,
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )

    def test_invalid_cluster_id_too_many(self):
        new_cluster_ids = np.append(self.clustering_prediction.cluster_ids, 1)
        with self.assertRaises(ValueError):
            dm.ClusteringPrediction(
                new_cluster_ids,
                self.clustering_prediction.costs,
                producer_id=dm.ProducerId("Test", "1.2.3"),
            )
