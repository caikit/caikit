"""
Tests for the Timeseries Evaluator data model object
"""
# Standard
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

# Standard
import json

# Third Party
from pandas.testing import assert_frame_equal
import numpy as np
import pandas as pd
import pytest

# Local
from caikit.core.data_model import ProducerId
import caikit.interfaces.ts.data_model as dm


@pytest.fixture(scope="module")
def basic_pandas_df():
    """Simple pandas df for testing target generation on multi-time series dataframe"""

    cv_pd_df_wo_offset = pd.DataFrame(
        {
            "prediction_mse": [0.013289, 0.010335, 1.099107],
            "prediction_mae": [0.115280, 0.101663, 0.841607],
            "prediction_smape": [0.010144, 0.006307, 0.014446],
        }
    )

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

    yield cv_pd_df_wo_offset, cv_pd_df_w_offset


@pytest.mark.parametrize("data", ["basic_pandas_df"], ids=["pandas"])
def test_Id_dm(data, request):
    cv_pd_df_wo_offset, cv_pd_df_w_offset = request.getfixturevalue(data)

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


@pytest.mark.parametrize("data", ["basic_pandas_df"], ids=["pandas"])
def test_EvaluationRecord_dm(data, request):
    cv_pd_df_wo_offset, cv_pd_df_w_offset = request.getfixturevalue(data)

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


@pytest.mark.parametrize("data", ["basic_pandas_df"], ids=["pandas"])
def test_EvaluationResult_w_offset(data, request):
    cv_pd_df_wo_offset, cv_pd_df_w_offset = request.getfixturevalue(data)

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


@pytest.mark.parametrize("data", ["basic_pandas_df"], ids=["pandas"])
def test_EvaluationResult_wo_offset(data, request):
    cv_pd_df_wo_offset, cv_pd_df_w_offset = request.getfixturevalue(data)

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


@pytest.mark.parametrize("data", ["basic_pandas_df"], ids=["pandas"])
def test_Errors(data, request):
    cv_pd_df_wo_offset, cv_pd_df_w_offset = request.getfixturevalue(data)

    id_values = cv_pd_df_w_offset["offset"].tolist()
    id_value_dms = [dm.Id(id_value) for id_value in id_values]

    sample_EvaluationRecords = cv_pd_df_w_offset.T.values.T.tolist()
    ER_dm = dm.EvaluationRecord(
        sample_EvaluationRecords[0][:2],
        sample_EvaluationRecords[0][2:5],
        sample_EvaluationRecords[0][5],
    )

    with pytest.raises(ValueError):
        dm.Id.from_proto(ER_dm.to_proto())
