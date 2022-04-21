from datetime import datetime

import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm

DATE_COLUMN_NAME = "Date"
DATE_INDEX_COLUMN_NAME = "DateIndex"
CSV_SUFFIX = ".csv"
PREPROCESS_SUFFIX = "_preprocessed"


def get_preprocessed_file_path(path: str) -> str:
    """XXX.csv => XXX_preprocessed.csv
    """
    return path.rstrip(CSV_SUFFIX) + PREPROCESS_SUFFIX + CSV_SUFFIX


def preprocess_file(input_path: str):
    output_path = get_preprocessed_file_path(input_path)

    df = pd.read_csv(input_path)
    date_list = list(df[DATE_COLUMN_NAME])
    date_index_list = [
        (parse(date, ignoretz=True) - datetime(1970, 1, 1)).days
        for date in tqdm(date_list, desc=f"Preprocessing {input_path}", total=df.shape[0])
    ]
    df[DATE_INDEX_COLUMN_NAME] = date_index_list
    df.drop([DATE_COLUMN_NAME], axis=1)
    df.sort_values(by=[DATE_INDEX_COLUMN_NAME])

    df.to_csv(output_path)
