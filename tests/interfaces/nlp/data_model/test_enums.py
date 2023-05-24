# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Third Party
from watson_nlp import data_model

# Local
# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestEnums(TestCaseBase):
    def test_forward_lookups(self):
        self.assertEqual(data_model.PartOfSpeech["POS_UNSET"], 0)
        self.assertEqual(data_model.PartOfSpeech.POS_UNSET, 0)

        self.assertEqual(data_model.DependencyRelation["DEP_OTHER"], 0)
        self.assertEqual(data_model.DependencyRelation.DEP_OTHER, 0)

    def test_reverse_lookups(self):
        self.assertEqual(data_model.PartOfSpeechRev[0], "POS_UNSET")
        self.assertEqual(
            data_model.PartOfSpeechRev[data_model.PartOfSpeech.POS_UNSET], "POS_UNSET"
        )

        self.assertEqual(data_model.DependencyRelationRev[0], "DEP_OTHER")
        self.assertEqual(
            data_model.DependencyRelationRev[
                data_model.DependencyRelation["DEP_OTHER"]
            ],
            "DEP_OTHER",
        )
