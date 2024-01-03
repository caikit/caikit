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
Tests for the Timeseries Evaluator data model object
"""
# Standard
import json
import warnings

# Third Party
from pandas.testing import assert_frame_equal
import pandas as pd
import pytest

# Local
from caikit.core.data_model import ProducerId
import caikit.interfaces.ts.data_model as dm

warnings.filterwarnings("ignore", category=ResourceWarning)


@pytest.fixture(scope="module")
def eval_df_wo_offset():
    """Simple pandas df for testing target generation on multi-time series dataframe"""

    cv_pd_df_wo_offset = pd.DataFrame(
        {
            "prediction_mse": [0.013289, 0.010335, 1.099107],
            "prediction_mae": [0.115280, 0.101663, 0.841607],
            "prediction_smape": [0.010144, 0.006307, 0.014446],
        }
    )

    yield cv_pd_df_wo_offset


@pytest.fixture(scope="module")
def eval_df_w_offset():
    cv_pd_df_w_offset = pd.DataFrame(
        {
            "ID1": ["A", "B", "C"],
            "ID2": ["D", "E", "F"],
            "prediction_mse": [0.013289, 0.010335, 1.099107],
            "prediction_mae": [0.115280, 0.101663, 0.841607],
            "prediction_smape": [0.010144, 0.006307, 0.014446],
            "offset": ["overall", 0, 1],
        }
    )

    yield cv_pd_df_w_offset


def test_Id_dm(eval_df_w_offset):
    cv_pd_df_w_offset = eval_df_w_offset

    id_values = cv_pd_df_w_offset["offset"].tolist()

    id_value_dms = [dm.Id(id_value) for id_value in id_values]

    id_value_dm = dm.Id.from_proto(id_value_dms[0].to_proto())
    assert id_value_dm.text == id_values[0]

    id_value_dm = dm.Id.from_proto(id_value_dms[1].to_proto())
    assert id_value_dm.index == id_values[1]

    id_value_dm = dm.Id.from_json(id_value_dms[0].to_json())
    assert id_value_dm.text == id_values[0]

    id_value_dm = dm.Id.from_json(json.loads(id_value_dms[0].to_json()))
    assert id_value_dm.text == id_values[0]

    id_value_dm = dm.Id.from_json(id_value_dms[2].to_json())
    assert id_value_dm.index == id_values[2]


def test_EvaluationRecord_dm(eval_df_wo_offset, eval_df_w_offset):
    cv_pd_df_wo_offset, cv_pd_df_w_offset = eval_df_wo_offset, eval_df_w_offset

    sample_EvaluationRecords = cv_pd_df_wo_offset.T.values.T.tolist()

    ER_dm = dm.EvaluationRecord(metric_values=sample_EvaluationRecords[1])

    assert len(ER_dm.id_values) == 0
    assert pytest.approx(ER_dm.metric_values[1]) == 0.101663
    assert ER_dm.offset == None

    ER_dm_rndTrip = dm.EvaluationRecord.from_proto(ER_dm.to_proto())
    assert len(ER_dm.id_values) == 0
    assert pytest.approx(ER_dm_rndTrip.metric_values[1]) == 0.101663
    assert ER_dm_rndTrip.offset == None

    ER_dm_rndTrip = dm.EvaluationRecord.from_json(ER_dm.to_json())
    assert len(ER_dm.id_values) == 0
    assert pytest.approx(ER_dm_rndTrip.metric_values[1]) == 0.101663
    assert ER_dm_rndTrip.offset == None

    sample_EvaluationRecords = cv_pd_df_w_offset.T.values.T.tolist()

    ER_dm = dm.EvaluationRecord(
        sample_EvaluationRecords[0][:2],
        sample_EvaluationRecords[0][2:5],
        sample_EvaluationRecords[0][5],
    )
    assert ER_dm.id_values[0].text == "A"
    assert ER_dm.id_values[1].text == "D"
    assert pytest.approx(ER_dm.metric_values[1]) == 0.11528
    assert ER_dm.offset.text == "overall"

    ER_dm_rndTrip = dm.EvaluationRecord.from_proto(ER_dm.to_proto())
    assert ER_dm_rndTrip.id_values[0].text == "A"
    assert ER_dm_rndTrip.id_values[1].text == "D"
    assert pytest.approx(ER_dm_rndTrip.metric_values[1]) == 0.11528
    assert ER_dm_rndTrip.offset.text == "overall"

    ER_dm_rndTrip = dm.EvaluationRecord.from_json(ER_dm.to_json())
    assert ER_dm_rndTrip.id_values[0].text == "A"
    assert ER_dm_rndTrip.id_values[1].text == "D"
    assert pytest.approx(ER_dm_rndTrip.metric_values[1]) == 0.11528
    assert ER_dm_rndTrip.offset.text == "overall"

    ER_dm = dm.EvaluationRecord(
        sample_EvaluationRecords[1][:2],
        sample_EvaluationRecords[1][2:5],
        sample_EvaluationRecords[1][5],
    )

    assert ER_dm.id_values[0].text == "B"
    assert ER_dm.id_values[1].text == "E"
    assert pytest.approx(ER_dm.metric_values[1]) == 0.101663
    assert ER_dm.offset.index == 0

    ER_dm_rndTrip = dm.EvaluationRecord.from_proto(ER_dm.to_proto())
    assert ER_dm_rndTrip.id_values[0].text == "B"
    assert ER_dm_rndTrip.id_values[1].text == "E"
    assert pytest.approx(ER_dm_rndTrip.metric_values[1]) == 0.101663
    assert ER_dm_rndTrip.offset.index == 0

    ER_dm_rndTrip = dm.EvaluationRecord.from_json(ER_dm.to_json())
    assert ER_dm_rndTrip.id_values[0].text == "B"
    assert ER_dm_rndTrip.id_values[1].text == "E"
    assert pytest.approx(ER_dm_rndTrip.metric_values[1]) == 0.101663
    assert ER_dm_rndTrip.offset.index == 0


def test_EvaluationResult_w_offset(eval_df_w_offset):
    cv_pd_df_w_offset = eval_df_w_offset

    ER_dm = dm.EvaluationResult(
        id_cols=["ID1", "ID2"],
        metric_cols=["prediction_mse", "prediction_mae", "prediction_smape"],
        offset_col="offset",
        df=cv_pd_df_w_offset,
        producer_id=ProducerId("Test", "1.0.0"),
    )
    assert ER_dm.metric_cols == ["prediction_mse", "prediction_mae", "prediction_smape"]
    assert ER_dm.id_cols == ["ID1", "ID2"]
    assert ER_dm.producer_id.name == "Test"
    assert ER_dm.producer_id.version == "1.0.0"

    pdf = ER_dm.as_pandas()
    assert_frame_equal(pdf, cv_pd_df_w_offset)
    assert (pdf.columns == cv_pd_df_w_offset.columns).all()
    assert pdf.columns.tolist() == [x for x in cv_pd_df_w_offset.columns]

    ER_dm_to_proto = ER_dm.to_proto()
    ER_dm_rndTrip = dm.EvaluationResult.from_proto(ER_dm_to_proto)
    assert ER_dm_rndTrip.records[0].id_values[0].text == "A"
    assert pytest.approx(ER_dm_rndTrip.records[0].metric_values[0]) == 0.013289
    assert ER_dm_rndTrip.id_cols[0] == "ID1"

    pdf = dm.EvaluationResult.from_proto(ER_dm.to_proto()).as_pandas()
    assert_frame_equal(pdf, cv_pd_df_w_offset)
    assert (pdf.columns == cv_pd_df_w_offset.columns).all()
    assert pdf.columns.tolist() == [x for x in cv_pd_df_w_offset.columns]

    ER_dm_to_json = json.loads(ER_dm.to_json())
    assert len(ER_dm_to_json["id_cols"]) == 2
    assert len(ER_dm_to_json["metric_cols"]) == 3
    assert ER_dm_to_json["offset_col"] == "offset"

    ER_dm_rndTrip = dm.EvaluationResult.from_json(ER_dm_to_json)
    assert ER_dm_rndTrip.records[0].id_values[0].text == "A"
    assert pytest.approx(ER_dm_rndTrip.records[0].metric_values[0]) == 0.013289
    assert ER_dm_rndTrip.id_cols[0] == "ID1"

    pdf = dm.EvaluationResult.from_json(ER_dm.to_json()).as_pandas()
    assert_frame_equal(pdf, cv_pd_df_w_offset)
    assert (pdf.columns == cv_pd_df_w_offset.columns).all()
    assert pdf.columns.tolist() == [x for x in cv_pd_df_w_offset.columns]


def test_EvaluationResult_wo_offset(eval_df_wo_offset):
    cv_pd_df_wo_offset = eval_df_wo_offset

    ER_dm = dm.EvaluationResult(
        metric_cols=["prediction_mse", "prediction_mae", "prediction_smape"],
        df=cv_pd_df_wo_offset,
    )

    assert ER_dm.metric_cols == ["prediction_mse", "prediction_mae", "prediction_smape"]
    assert len(ER_dm.id_cols) == 0

    pdf = ER_dm.as_pandas()
    assert_frame_equal(pdf, cv_pd_df_wo_offset)
    assert (pdf.columns == cv_pd_df_wo_offset.columns).all()
    assert pdf.columns.tolist() == [x for x in cv_pd_df_wo_offset.columns]

    ER_dm_to_proto = ER_dm.to_proto()
    ER_dm_rndTrip = dm.EvaluationResult.from_proto(ER_dm_to_proto)
    assert len(ER_dm_rndTrip.records[0].id_values) == 0
    assert pytest.approx(ER_dm_rndTrip.records[0].metric_values[0]) == 0.013289
    assert len(ER_dm_rndTrip.id_cols) == 0

    pdf = dm.EvaluationResult.from_proto(ER_dm.to_proto()).as_pandas()
    assert_frame_equal(pdf, cv_pd_df_wo_offset)
    assert (pdf.columns == cv_pd_df_wo_offset.columns).all()
    assert pdf.columns.tolist() == [x for x in cv_pd_df_wo_offset.columns]

    ER_dm_to_json = json.loads(ER_dm.to_json())
    assert len(ER_dm_to_json["id_cols"]) == 0
    assert len(ER_dm_to_json["metric_cols"]) == 3
    assert ER_dm_to_json["offset_col"] == None

    ER_dm_rndTrip = dm.EvaluationResult.from_json(ER_dm_to_json)
    assert len(ER_dm_rndTrip.records[0].id_values) == 0
    assert pytest.approx(ER_dm_rndTrip.records[0].metric_values[0]) == 0.013289
    assert len(ER_dm_rndTrip.id_cols) == 0

    pdf = dm.EvaluationResult.from_json(ER_dm.to_json()).as_pandas()
    assert_frame_equal(pdf, cv_pd_df_wo_offset)
    assert (pdf.columns == cv_pd_df_wo_offset.columns).all()
    assert pdf.columns.tolist() == [x for x in cv_pd_df_wo_offset.columns]


def test_Errors(eval_df_w_offset):
    cv_pd_df_w_offset = eval_df_w_offset

    id_values = cv_pd_df_w_offset["offset"].tolist()
    # id_value_dms = [dm.Id(id_value) for id_value in id_values]

    sample_EvaluationRecords = cv_pd_df_w_offset.T.values.T.tolist()
    ER_dm = dm.EvaluationRecord(
        sample_EvaluationRecords[0][:2],
        sample_EvaluationRecords[0][2:5],
        sample_EvaluationRecords[0][5],
    )

    with pytest.raises(ValueError):
        dm.Id.from_proto(ER_dm.to_proto())
