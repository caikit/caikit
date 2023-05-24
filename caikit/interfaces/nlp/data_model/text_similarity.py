# *****************************************************************#
# (C) Copyright IBM Corporation 2022.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#
"""Data structures for text similarity predictions.
"""
# Standard
from typing import List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit import error_handler
from ...common.data_model import ProducerId

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="watson_core_data_model.nlp")
class TextSimilarityPrediction(DataObjectBase):
    scores: Annotated[List[float], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a similarity scores prediction."""

    def __init__(self, scores, producer_id=None):
        """Construct a new similarity scores prediction.

        Args:
            scores:  list(float)
                The similarity scores associated with this
                similarity prediction.
            producer_id:  ProducerId or None
                The block that produced this similarity prediction.
        """
        error.type_check("<NLP57445742E>", list, scores=scores)
        error.type_check(
            "<NLP85436555E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.scores = scores[:]
        self.producer_id = producer_id
