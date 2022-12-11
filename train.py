import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==0.23.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib==3.2.1"])

import glob
import time
import random
from glob import glob
import argparse
import json
import os

from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


CLASSES = [1, 2, 3, 4, 5]


def select_data_and_label_from_record(record):
    x = {"input_ids": record["input_ids"], "input_mask": record["input_mask"], "segment_ids": record["segment_ids"]}
    y = record["label_ids"]
    return (x, y)


def file_based_input_dataset_builder(
    channel,
    input_filenames,
    pipe_mode,
    is_training,
    drop_remainder,
    batch_size,
    epochs,
    steps_per_epoch,
    max_seq_length,
):
    if pipe_mode:
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel, record_format="TFRecord")
    else:
        dataset = tf.data.TFRecordDataset(input_filenames)

    dataset = dataset.repeat(epochs * steps_per_epoch * 100)
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        record = tf.io.parse_single_example(record, name_to_features)
        return record

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    row_count = 0
    for row in dataset.as_numpy_iterator():
        print(row)
        if row_count == 5:
            break
        row_count = row_count + 1

    return dataset


def load_checkpoint_model(checkpoint_path):
    glob_pattern = os.path.join(checkpoint_path, "*.h5")
    list_of_checkpoint_files = glob.glob(glob_pattern)
    latest_checkpoint_file = max(list_of_checkpoint_files)
    initial_epoch_number_str = latest_checkpoint_file.rsplit("_", 1)[-1].split(".h5")[0]
    initial_epoch_number = int(initial_epoch_number_str)
    loaded_model = TFDistilBertForSequenceClassification.from_pretrained(latest_checkpoint_file, config=config)
    return loaded_model, initial_epoch_number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--use_xla", type=eval, default=False)
    parser.add_argument("--use_amp", type=eval, default=False)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--validation_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00003)
    parser.add_argument("--epsilon", type=float, default=0.00000001)
    parser.add_argument("--train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--test_steps", type=int, default=None)
    parser.add_argument("--freeze_bert_layer", type=eval, default=False)
    parser.add_argument("--enable_sagemaker_debugger", type=eval, default=False)
    parser.add_argument("--run_validation", type=eval, default=False)
    parser.add_argument("--run_test", type=eval, default=False)
    parser.add_argument("--run_sample_predictions", type=eval, default=False)
    parser.add_argument("--enable_tensorboard", type=eval, default=False)
    parser.add_argument("--enable_checkpointing", type=eval, default=False)
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])  # This is unused

    args, _ = parser.parse_known_args()
    env_var = os.environ

    sm_training_env_json = json.loads(env_var["SM_TRAINING_ENV"])
    is_master = sm_training_env_json["is_master"]

    train_data = args.train_data
    validation_data = args.validation_data
    test_data = args.test_data
    local_model_dir = os.environ["SM_MODEL_DIR"]
    output_dir = args.output_dir
    hosts = args.hosts
    current_host = args.current_host
    num_gpus = args.num_gpus
    job_name = os.environ["SAGEMAKER_JOB_NAME"]
    use_xla = args.use_xla
    use_amp = args.use_amp
    max_seq_length = args.max_seq_length
    train_batch_size = args.train_batch_size
    validation_batch_size = args.validation_batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    epsilon = args.epsilon
    train_steps_per_epoch = args.train_steps_per_epoch
    validation_steps = args.validation_steps
    test_steps = args.test_steps
    freeze_bert_layer = args.freeze_bert_layer
    enable_sagemaker_debugger = args.enable_sagemaker_debugger
    run_validation = args.run_validation
    run_test = args.run_test
    run_sample_predictions = args.run_sample_predictions
    enable_tensorboard = args.enable_tensorboard
    enable_checkpointing = args.enable_checkpointing
    checkpoint_base_path = args.checkpoint_base_path

    if is_master:
        checkpoint_path = checkpoint_base_path
    else:
        checkpoint_path = "/tmp/checkpoints"

    pipe_mode_str = os.environ.get("SM_INPUT_DATA_CONFIG", "")
    pipe_mode = pipe_mode_str.find("Pipe") >= 0

    transformer_fine_tuned_model_path = os.path.join(local_model_dir, "transformers/fine-tuned/")
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)

    tensorflow_saved_model_path = os.path.join(local_model_dir, "tensorflow/saved_model/0")
    os.makedirs(tensorflow_saved_model_path, exist_ok=True)

    tensorboard_logs_path = os.path.join(local_model_dir, "tensorboard/")
    os.makedirs(tensorboard_logs_path, exist_ok=True)

    distributed_strategy = tf.distribute.MirroredStrategy()

    with distributed_strategy.scope():
        tf.config.optimizer.set_jit(use_xla)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": use_amp})

        train_data_filenames = glob(os.path.join(train_data, "*.tfrecord"))
        train_dataset = file_based_input_dataset_builder(
            channel="train",
            input_filenames=train_data_filenames,
            pipe_mode=pipe_mode,
            is_training=True,
            drop_remainder=False,
            batch_size=train_batch_size,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            max_seq_length=max_seq_length,
        ).map(select_data_and_label_from_record)

        tokenizer = None
        config = None
        model = None
        transformer_model = None

        successful_download = False
        retries = 0
        while retries < 5 and not successful_download:
            try:
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                config = DistilBertConfig.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=len(CLASSES),
                    id2label={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
                    label2id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
                )

                transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", config=config
                )

                input_ids = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids", dtype="int32")
                input_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_mask", dtype="int32")

                embedding_layer = transformer_model.distilbert(input_ids, attention_mask=input_mask)[0]
                X = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
                )(embedding_layer)
                X = tf.keras.layers.GlobalMaxPool1D()(X)
                X = tf.keras.layers.Dense(50, activation="relu")(X)
                X = tf.keras.layers.Dropout(0.2)(X)
                X = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(X)

                model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=X)

                for layer in model.layers[:3]:
                    layer.trainable = not freeze_bert_layer

                successful_download = True
                print("Sucessfully downloaded after {} retries.".format(retries))
            except:
                retries = retries + 1
                random_sleep = random.randint(1, 30)
                print("Retry #{}.  Sleeping for {} seconds".format(retries, random_sleep))
                time.sleep(random_sleep)

        callbacks = []

        initial_epoch_number = 0

        if enable_checkpointing:
            os.makedirs(checkpoint_path, exist_ok=True)
            if os.listdir(checkpoint_path):
                model, initial_epoch_number = load_checkpoint_model(checkpoint_path)

            checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(checkpoint_path, "tf_model_{epoch:05d}.h5"),
                save_weights_only=False,
                verbose=1,
                monitor="val_accuracy",
            )
            callbacks.append(checkpoint_callback)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)

        if use_amp:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

        if enable_tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs_path)
            callbacks.append(tensorboard_callback)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        if run_validation:
            validation_data_filenames = glob(os.path.join(validation_data, "*.tfrecord"))
            validation_dataset = file_based_input_dataset_builder(
                channel="validation",
                input_filenames=validation_data_filenames,
                pipe_mode=pipe_mode,
                is_training=False,
                drop_remainder=False,
                batch_size=validation_batch_size,
                epochs=epochs,
                steps_per_epoch=validation_steps,
                max_seq_length=max_seq_length,
            ).map(select_data_and_label_from_record)

            validation_dataset = validation_dataset.take(validation_steps)
            train_and_validation_history = model.fit(
                train_dataset,
                shuffle=True,
                epochs=epochs,
                initial_epoch=initial_epoch_number,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=validation_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
            )
        else:
            train_history = model.fit(
                train_dataset,
                shuffle=True,
                epochs=epochs,
                initial_epoch=initial_epoch_number,
                steps_per_epoch=train_steps_per_epoch,
                callbacks=callbacks,
            )

        if run_test:
            test_data_filenames = glob(os.path.join(test_data, "*.tfrecord"))
            test_dataset = file_based_input_dataset_builder(
                channel="test",
                input_filenames=test_data_filenames,
                pipe_mode=pipe_mode,
                is_training=False,
                drop_remainder=False,
                batch_size=test_batch_size,
                epochs=epochs,
                steps_per_epoch=test_steps,
                max_seq_length=max_seq_length,
            ).map(select_data_and_label_from_record)

            test_history = model.evaluate(test_dataset, steps=test_steps, callbacks=callbacks)

        transformer_model.save_pretrained(transformer_fine_tuned_model_path)
        model.save(tensorflow_saved_model_path, include_optimizer=False, overwrite=True, save_format="tf")

        inference_path = os.path.join(local_model_dir, "code/")
        os.makedirs(inference_path, exist_ok=True)
        os.system("cp inference.py {}".format(inference_path))
        os.system("cp -R ./test_data/ {}".format(local_model_dir))
