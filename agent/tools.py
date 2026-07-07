import pandas as pd

DATA_PATH = "data/experiment_runs.csv"
_df = None


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(path)
    return _df


def get_columns() -> list:
    return list(load_data().columns)


VALID_OPERATIONS = ["mean", "max", "min", "sum", "count", "median", "std"]


def query_dataset(operation: str, column: str, filter_column: str = None,
                   filter_value=None, group_by: str = None) -> dict:
    df = load_data()

    if operation not in VALID_OPERATIONS:
        return {"success": False,
                "error": f"'{operation}' is not supported. Choose from: {VALID_OPERATIONS}."}

    if column not in df.columns and operation != "count":
        return {"success": False,
                "error": f"Column '{column}' does not exist. Available: {get_columns()}."}

    working = df

    if filter_column is not None:
        if filter_column not in df.columns:
            return {"success": False,
                    "error": f"Filter column '{filter_column}' does not exist. Available: {get_columns()}."}
        working = working[working[filter_column] == filter_value]
        if working.empty:
            return {"success": False,
                    "error": f"No rows match {filter_column} == {filter_value!r}."}

    try:
        if group_by is not None:
            if group_by not in df.columns:
                return {"success": False,
                        "error": f"Group-by column '{group_by}' does not exist. Available: {get_columns()}."}
            if operation == "count":
                result = working.groupby(group_by).size().to_dict()
            else:
                raw = getattr(working.groupby(group_by)[column], operation)().to_dict()
                result = {k: round(float(v), 4) for k, v in raw.items()}
            return {"success": True, "operation": operation, "column": column,
                    "group_by": group_by, "result": result, "row_count": int(len(working))}
        else:
            if operation == "count":
                value = int(len(working))
            else:
                value = round(float(getattr(working[column], operation)()), 4)
            return {"success": True, "operation": operation, "column": column,
                    "result": value, "row_count": int(len(working))}
    except Exception as e:
        return {"success": False, "error": f"Computation failed: {e}"}