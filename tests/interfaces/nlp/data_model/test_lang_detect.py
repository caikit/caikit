# *****************************************************************#
# (C) Copyright IBM Corporation 2020.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#

# Third Party
import utils

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestLangDetect(TestCaseBase):
    def setUp(self):
        self.lang_detect = dm.LangDetectPrediction(
            lang_code=dm.enums.LangCode["LANG_EN"]
        )

    def test_fields(self):
        self.assertTrue(utils.validate_fields(self.lang_detect))

    def test_from_proto_and_back(self):
        new = dm.LangDetectPrediction.from_proto(self.lang_detect.to_proto())
        self.assertEqual(new.lang_code, self.lang_detect.lang_code)

    def test_from_json_and_back(self):
        new = dm.LangDetectPrediction.from_json(self.lang_detect.to_json())
        self.assertEqual(new.lang_code, self.lang_detect.lang_code)

    def test_to_string(self):
        self.assertEqual(self.lang_detect.to_string(), "LANG_EN")

    def test_to_iso_format(self):
        self.assertEqual(self.lang_detect.to_iso_format(), "EN")
