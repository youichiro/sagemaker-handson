import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "anaconda", "tensorflow==2.3.0", "-y"])
subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "transformers==3.5.1", "-y"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])

import os
import argparse
import json
import os
import numpy as np
import csv
import glob
import tarfile
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import DistilBertTokenizer, DistilBertConfig
from sklearn.metrics import confusion_matrix, accuracy_score


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

CLASSES = [1, 2, 3, 4, 5]

config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(CLASSES),
    id2label={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    label2id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
)


def list_arg(raw_value):
    return str(raw_value).split(",")


def predict(text, model, max_seq_length):
    encode_plus_tokens = tokenizer.encode_plus(
        text, pad_to_max_length=True, max_length=max_seq_length, truncation=True, return_tensors="tf"
    )
    input_ids = encode_plus_tokens["input_ids"]
    input_mask = encode_plus_tokens["attention_mask"]
    outputs = model.predict(x=(input_ids, input_mask))
    prediction = [{"label": config.id2label[item.argmax()], "score": item.max().item()} for item in outputs]
    return prediction[0]["label"]


def process(args):
    model_tar_path = "{}/model.tar.gz".format(args.input_model)
    model_tar = tarfile.open(model_tar_path)
    model_tar.extractall(args.input_model)
    model_tar.close()

    model = keras.models.load_model("{}/tensorflow/saved_model/0".format(args.input_model))
    test_data_path = "{}/amazon_reviews_us_Digital_Software_v1_00.tsv.gz".format(args.input_data)

    df_test_reviews = pd.read_csv(test_data_path, delimiter="\t", quoting=csv.QUOTE_NONE, compression="gzip")[
        ["review_body", "star_rating"]
    ]

    df_test_reviews = df_test_reviews.sample(n=100)
    df_test_reviews.shape
    df_test_reviews.head()

    y_test = [predict(review, model, args.max_seq_length) for review in df_test_reviews["review_body"]]
    y_actual = df_test_reviews["star_rating"]
    accuracy = accuracy_score(y_true=y_test, y_pred=y_actual)

    def plot_conf_mat(cm, classes, title, cmap):
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="black" if cm[i, j] > thresh else "black",
            )

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

    cm = confusion_matrix(y_true=y_test, y_pred=y_actual)

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_conf_mat(cm, classes=CLASSES, title="Confusion Matrix", cmap=plt.cm.Greens)
    plt.show()

    metrics_path = os.path.join(args.output_data, "metrics/")
    os.makedirs(metrics_path, exist_ok=True)
    plt.savefig("{}/confusion_matrix.png".format(metrics_path))
    report_dict = {
        "metrics": {
            "accuracy": {
                "value": accuracy,
            },
        },
    }

    evaluation_path = "{}/evaluation.json".format(metrics_path)
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    print("Complete")


def parse_args():
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")

    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        default="/opt/ml/processing/input/model",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args)
