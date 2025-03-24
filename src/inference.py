import argparse
import polars as pl
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    pipeline,
    MBartForConditionalGeneration,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--test", type=str, default=None, help="Path to test data")
args = parser.parse_args()

assert args.ckpt is not None

MAX_LENGTH = 256
MODEL_NAME = "vinai/bartpho-syllable"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(args.ckpt)
model.eval().cuda()

corrector = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cuda",
)

# Define the text samples
data = pl.read_csv(args.test)
texts = data["text"].to_list()
corrected_texts = data["summary"].to_list()

# Batch prediction
predictions = corrector(texts, max_length=MAX_LENGTH)


def calculate_cer(pred, target):
    pred = pred.replace(" ", "")
    target = target.replace(" ", "")
    cer = 0
    for i in range(len(pred)):
        if i < len(target):
            if pred[i] != target[i]:
                cer += 1
        else:
            cer += 1

    return cer / len(target)


cer = 0
total_cer = 0

for i, (text, pred) in tqdm(enumerate(zip(texts, predictions)), total=len(texts)):
    total_cer += calculate_cer(
        pred["generated_text"].strip(), corrected_texts[i].strip()
    )

    print(text)
    print(pred["generated_text"])
    print(corrected_texts[i])
    print("---" * 20)

print(f"Average CER: {total_cer / len(texts)}")
