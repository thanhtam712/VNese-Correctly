import argparse
import polars as pl
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-csv", type=str, help="Path to csv file")
    return parser.parse_args()


def read():
    """
    Read csv file by using polars
    """
    args = parse_args()
    df = pl.read_csv(args.file_csv)
    for data in tqdm(df.dicts(), total=len(df), desc="Reading csv file"):
        print(data)

    return df


if __name__ == "__main__":
    read()
