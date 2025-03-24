import polars as pl
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def read_csv(input_files: str) -> pl.DataFrame:
    """
    Read csv file by using polars

    Args:
        input_files: str

    Returns:
        pl.DataFrame
    """
    df = pl.read_csv(input_files)
    header = df.columns
    df = df.rename({header[0]: "input", header[1]: "output"})
    return df


def train_test_datasets(train_files: list[str]) -> DatasetDict:
    """
    Read train and test data from csv files and convert to Dataset type

    Returns:
        DatasetDict
    """
    list_train_df = []
    for f_input in tqdm(train_files, total=len(train_files), desc="Reading train data"):
        train_df = read_csv(f_input)
        train_df = train_df.to_dicts()
        list_train_df.extend(train_df)

    # list_test_df = []
    # for f_input in tqdm(test_files, total=len(test_files), desc="Reading test data"):
    #     test_df = read_csv(f_input)
    #     test_df = test_df.to_dicts()
    #     list_test_df.extend(test_df)

    return DatasetDict(
        {
            "train": Dataset.from_list(list_train_df),
            # "test": Dataset.from_list(list_test_df),
        }
    )
