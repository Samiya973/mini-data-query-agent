import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from agent.tools import query_dataset, get_columns


def test_mean_matches_pandas_directly():
    df = pd.read_csv("data/experiment_runs.csv")
    expected = round(float(df["roc_auc"].mean()), 4)
    result = query_dataset("mean", "roc_auc")
    assert result["success"] is True
    assert result["result"] == expected


def test_max_matches_pandas_directly():
    df = pd.read_csv("data/experiment_runs.csv")
    expected = round(float(df["roc_auc"].max()), 4)
    result = query_dataset("max", "roc_auc")
    assert result["result"] == expected


def test_count_all_rows():
    df = pd.read_csv("data/experiment_runs.csv")
    result = query_dataset("count", "roc_auc")
    assert result["result"] == len(df)


def test_filter_narrows_correctly():
    df = pd.read_csv("data/experiment_runs.csv")
    subset = df[df["top_shap_feature"] == "annual_income"]
    expected = round(float(subset["roc_auc"].mean()), 4)
    result = query_dataset(
        "mean", "roc_auc",
        filter_column="top_shap_feature", filter_value="annual_income",
    )
    assert result["result"] == expected
    assert result["row_count"] == len(subset)


def test_group_by_covers_all_groups():
    df = pd.read_csv("data/experiment_runs.csv")
    result = query_dataset("mean", "roc_auc", group_by="top_shap_feature")
    assert set(result["result"].keys()) == set(df["top_shap_feature"].unique())


def test_nonexistent_column_returns_structured_error_not_crash():
    result = query_dataset("mean", "this_column_does_not_exist")
    assert result["success"] is False
    assert "does not exist" in result["error"]


def test_nonexistent_operation_returns_structured_error():
    result = query_dataset("multiply_by_seven", "roc_auc")
    assert result["success"] is False
    assert "is not supported" in result["error"]


def test_filter_with_no_matches_returns_structured_error():
    result = query_dataset(
        "mean", "roc_auc",
        filter_column="top_shap_feature", filter_value="not_a_real_feature",
    )
    assert result["success"] is False


def test_get_columns_matches_real_csv_header():
    df = pd.read_csv("data/experiment_runs.csv")
    assert get_columns() == list(df.columns)