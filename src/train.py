import warnings
import evaluate
import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

from prepare_dataset import train_test_datasets

warnings.filterwarnings("ignore")

MAX_LENGTH = 256
MODEL_NAME = "vinai/bartpho-syllable"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-train", type=str, nargs="+", help="List path to train data"
    )
    # parser.add_argument(
    #     "--input-test",
    #     type=str,
    #     nargs="+",
    #     required=False,
    #     help="List path to test data",
    # )
    return parser.parse_args()


def preprocess_function(examples):
    return tokenizer(
        examples["input"],
        text_target=examples["output"],
        max_length=MAX_LENGTH,
        truncation=True,
    )


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"sacrebleu": result["score"]}


args = parse_args()
dataset = train_test_datasets(train_files=args.input_train)
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=8,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = evaluate.load("sacrebleu")

args = Seq2SeqTrainingArguments(
    do_train=True,
    eval_strategy="no",
    output_dir="output",
    num_train_epochs=1,
    learning_rate=1e-5,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    logging_steps=20_000,
    save_total_limit=5,
    predict_with_generate=True,
    fp16=True,
    ddp_find_unused_parameters=False,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# trainer.evaluate()

# trainer.push_to_hub(tags="vnese-correctly-nlp", commit_message="Training complete")
