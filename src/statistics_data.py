import argparse
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import DatasetDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-train", type=str, nargs="+", help="List path to train data"
    )
    parser.add_argument(
        "--input-test",
        type=str,
        nargs="+",
        required=False,
        help="List path to test data",
    )
    return parser.parse_args()


def read_csv(input_files: str) -> tuple[int, int, int, int, set[str]]:
    """
    Read csv file by using polars

    Args:
        input_files: str

    Returns:
        max_length: int
        min_length: int
        sum_length: int
        len: int
    """
    df = pl.read_csv(input_files)
    max_length, min_length, sum_length = 0, -1, 0

    header = df.columns
    df = df.rename({header[0]: "input", header[1]: "output"})

    max_length = (
        df["output"].map_elements(lambda x: len(x.split()), return_dtype=pl.Int64).max()
    )
    min_length = (
        df["output"].map_elements(lambda x: len(x.split()), return_dtype=pl.Int64).min()
    )
    sum_length = (
        df["output"].map_elements(lambda x: len(x.split()), return_dtype=pl.Int64).sum()
    )

    vocab = set()
    df["output"].map_elements(lambda x: vocab.update(x.split()))

    return max_length, min_length, sum_length, len(df), vocab


def train_test_datasets(train_files: list[str], test_files: list[str]) -> DatasetDict:
    """
    Read train and test data from csv files and convert to Dataset type

    Returns:
        DatasetDict
    """
    max_train, min_train, sum_train, len_train = 0, -1, 0, 0
    vocab_train = set()
    for f_input in tqdm(train_files, total=len(train_files), desc="Reading train data"):
        max_f, min_f, sum_f, len_f, vocab_train_f = read_csv(f_input)
        vocab_train.update(vocab_train_f)
        max_train = max(max_train, max_f)
        if min_train != -1:
            min_train = min(min_train, min_f)
        else:
            min_train = min_f
        sum_train += sum_f
        len_train += len_f

    max_test, min_test, sum_test, len_test = 0, -1, 0, 0
    vocab_test = set()
    for f_input in tqdm(test_files, total=len(test_files), desc="Reading test data"):
        max_f, min_f, sum_f, len_f, vocab_test_f = read_csv(f_input)
        vocab_test.update(vocab_test_f)
        max_test = max(max_test, max_f)
        if min_test != -1:
            min_test = min(min_test, min_f)
        else:
            min_test = min_f
        sum_test += sum_f
        len_test += len_f

    results = {
        "train": (max_train, min_train, sum_train, len_train, len(vocab_train)),
        "test": (max_test, min_test, sum_test, len_test, len(vocab_test)),
    }
    return results


def visualize_data(results: dict) -> None:
    """
    Visualize the data

    Args:
        results: dict
    """

    categories = ["Max Length", "Min Length", "Avg Length"]
    train_values = [
        results["train"][0],
        results["train"][1],
        results["train"][2] / results["train"][3],
    ]
    test_values = [
        results["test"][0],
        results["test"][1],
        results["test"][2] / results["test"][3],
    ]

    data_train = {
        "Category": categories,
        "Value": train_values,
        "Dataset": ["Train_0", "Train_1", "Train_2"],
    }

    data_test = {
        "Category": categories,
        "Value": test_values,
        "Dataset": ["Test_0", "Test_1", "Test_2"],
    }

    plt.figure(figsize=(8, 6))
    ax_train = sns.barplot(x="Category", y="Value", hue="Dataset", data=data_train)
    for p in ax_train.patches:
        ax_train.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.title("Comparison of Max, Min and Avg Length in Train Set")
    plt.ylabel("Word Count")
    plt.legend(title="Dataset")
    plt.savefig("train.png")

    plt.figure(figsize=(8, 6))
    ax_test = sns.barplot(x="Category", y="Value", hue="Dataset", data=data_test)
    for p in ax_test.patches:
        ax_test.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.title("Comparison of Max, Min and Avg Length in Test Set")
    plt.ylabel("Word Count")
    plt.legend(title="Dataset")
    plt.savefig("test.png")


args = parse_args()
results = train_test_datasets(train_files=args.input_train, test_files=args.input_test)
print(results)
visualize_data(results)
