# *****************************************************************#
# (C) Copyright IBM Corporation 2023.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Local
from . import utils
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class GeneratedResult(TestCaseBase):
    def setUp(self):
        self.generation_prediction = dm.GeneratedResult(
            text="It is 20 degrees today", stop_reason=1, generated_token_count=100
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.generation_prediction))

    def test_from_proto_and_back(self):
        new = dm.GeneratedResult.from_proto(self.generation_prediction.to_proto())
        self.assertEqual(new.text, self.generation_prediction.text)
        self.assertEqual(new.stop_reason, self.generation_prediction.stop_reason)
        self.assertEqual(
            new.generated_token_count, self.generation_prediction.generated_token_count
        )

    def test_from_json_and_back(self):
        new = dm.GeneratedResult.from_json(self.generation_prediction.to_json())
        self.assertEqual(new.text, self.generation_prediction.text)
        self.assertEqual(new.stop_reason, self.generation_prediction.stop_reason)
        self.assertEqual(
            new.generated_token_count, self.generation_prediction.generated_token_count
        )
