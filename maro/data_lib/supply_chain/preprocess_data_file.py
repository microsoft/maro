from datetime import datetime

import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm

DATE_INDEX_COLUMN_NAME = "DateIndex"
CSV_SUFFIX = ".csv"
PREPROCESS_SUFFIX = "_preprocessed"


def get_preprocessed_file_path(path: str) -> str:
    """XXX.csv => XXX_preprocessed.csv"""
    return path.rstrip(CSV_SUFFIX) + PREPROCESS_SUFFIX + CSV_SUFFIX


def get_date_index(date: datetime) -> int:
    return (date - datetime(1970, 1, 1)).days


def preprocess_file(input_path: str, date_column_name: str) -> None:
    output_path = get_preprocessed_file_path(input_path)

    df = pd.read_csv(input_path)
    date_list = list(df[date_column_name])
    date_index_list = [
        get_date_index(parse(date, ignoretz=True))
        for date in tqdm(date_list, desc=f"Preprocessing {input_path}", total=df.shape[0])
    ]
    df[DATE_INDEX_COLUMN_NAME] = date_index_list
    df.drop([date_column_name], axis=1, inplace=True)
    df.sort_values(by=[DATE_INDEX_COLUMN_NAME], inplace=True)

    df.to_csv(output_path)
