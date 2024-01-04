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
"""
The core data model object for a TimeSeries Evaluator.
"""
# Standard
from typing import List, Union

# Third Party
import pandas as pd

# First Party
from py_to_proto.dataclass_to_proto import (  # Annotated imported from here for compatibility
    Annotated,
    FieldNumber,
    OneofField,
)
import alog

# Local
from ....core import DataObjectBase
from ....core.data_model import ProducerId, dataobject
from ....core.exceptions import error_handler
from .package import TS_PACKAGE

log = alog.use_channel("TSEDM")
error = error_handler.get(log)

## TimeSeries Evaluator ##################################################################


@dataobject(package=TS_PACKAGE)
class Id(DataObjectBase):
    """A single instance of Id
    Representation of ids that can be either text or index. Customized
    this way to be able to work with repeated
    """

    value: Union[
        Annotated[str, OneofField("text"), FieldNumber(1)],
        Annotated[int, OneofField("index"), FieldNumber(2)],
    ]


@dataobject(package=TS_PACKAGE)
class EvaluationRecord(DataObjectBase):
    """A single EvaluationRecord for EvaluationResult
    Representation of EvaluationRecord for each row in the dataframe
    EvaluationRecord{id_values=["A", "B"], metric_values=[0.234, 0.568, 0.417], offset="overall"}
    """

    id_values: Annotated[List[Id], FieldNumber(1)]
    metric_values: Annotated[List[float], FieldNumber(2)]
    offset: Annotated[Id, FieldNumber(3)]

    def __init__(self, id_values=None, metric_values=None, offset=None):
        """Construct a new EvaluationRecord instance

        EvaluationRecord

        Args:
            id_values: list(Id)
                List of Id values for the record
            metric_values: list(float)
                List of Id values containing metric results for the record
            offset: (optional) Id
                offset associated with the record
        """

        error.type_check_all(
            "<COR26895394E>", str, int, Id, allow_none=True, id_values=id_values
        )
        error.type_check_all("<COR25875394E>", float, metric_values=metric_values)
        error.type_check("<COR25873484E>", str, int, Id, allow_none=True, offset=offset)

        super().__init__()

        self.id_values = (
            []
            if id_values is None
            else [
                Id(id_value) if not isinstance(id_value, Id) else id_value
                for id_value in id_values
            ]
        )

        self.metric_values = metric_values

        self.offset = (
            None
            if offset is None
            else Id(offset)
            if not isinstance(offset, Id)
            else offset
        )


@dataobject(package=TS_PACKAGE)
class EvaluationResult(DataObjectBase):
    """EvaluationResult containing the evaluation results
    Representation of EvaluationResult stores rows of the dataframe as list of records string lists
    to keep track of id and metric columns
    """

    records: Annotated[List[EvaluationRecord], FieldNumber(1)]
    id_cols: Annotated[List[str], FieldNumber(2)]
    metric_cols: Annotated[List[str], FieldNumber(3)]
    offset_col: Annotated[str, FieldNumber(4)]
    producer_id: Annotated[ProducerId, FieldNumber(5)]

    def __init__(
        self,
        records=None,
        id_cols=None,
        metric_cols=None,
        offset_col=None,
        df=None,
        producer_id=None,
    ):
        """Construct a new EvaluationResult instance

        EvaluationResult

        Args:
            records: list(EvaluationRecord)
                List of Evaluation Record instances
            id_cols: list(string)
                List of string containing id column names (Optional)
            metric_cols: list(string)
                List of string containing metric value column names
            offset_col: string
                Name of offset column in dataframe if exists (Optional)
            df: pandas dataframe
                initial input dataframe from which to store the results
            producer_id:  ProducerId | None
                The module that produced this evaluation result.
        """

        error.type_check_all("<COR25782594E>", str, allow_none=True, id_cols=id_cols)
        error.type_check_all("<COR28634484E>", str, metric_cols=metric_cols)
        error.type_check("<COR28485384E>", str, allow_none=True, offset_col=offset_col)
        error.type_check(
            "<COR28485385E>",
            tuple,
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()

        self.id_cols = [] if id_cols is None else id_cols
        self.metric_cols = metric_cols
        self.offset_col = offset_col
        self.producer_id = producer_id

        if df is not None:
            if self.offset_col is not None:
                error.value_check(
                    "<COR28484474E>",
                    self.offset_col in df.columns,
                    f"Specified '{self.offset_col}' offset column not in dataframe",
                )

            self.records = [
                EvaluationRecord(
                    id_values=(
                        None
                        if len(self.id_cols) == 0
                        else df.loc[i, self.id_cols].values.tolist()
                    ),
                    metric_values=df.loc[i, self.metric_cols].values.tolist(),
                    offset=(
                        None if self.offset_col is None else df.loc[i, self.offset_col]
                    ),
                )
                for i in range(len(df))
            ]
        else:
            error.type_check_all("<COR32696407E>", EvaluationRecord, records=records)
            self.records = records

    def as_pandas(self) -> "pd.DataFrame":
        """Generate and return a pandas DataFrame"""

        records = []

        has_offset = False
        for record in self.records:
            id_values = []
            metric_values = []
            offset = None

            id_values = [v.value for v in record.id_values]
            metric_values = record.metric_values
            if record.offset:
                offset = record.offset.value
                has_offset = True

            records.append(id_values + metric_values + [offset])

        df = pd.DataFrame(
            records, columns=self.id_cols + self.metric_cols + [self.offset_col]
        )
        if not has_offset:
            df.drop([self.offset_col], axis=1, inplace=True)

        return df
