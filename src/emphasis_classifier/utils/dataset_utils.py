# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import evaluate
import numpy as np
import ast

def add_label(examples):
    tensors = []
    for emphass_str in examples["emphasis"]:
        nums = list(map(int, emphass_str.split()))
        tensor = torch.tensor(nums)
        tensors.append(tensor)
    return {"labels": tensors}



def preprocess_function(examples, feature_extractor, num_labels):
    # Get audio arrays
    audio_arrays = [x["array"] for x in examples["audio"]]

    # Convert stereo audio to mono if necessary
    audio_arrays = [torch.mean(audio, dim=0) if len(audio.shape) > 1 and audio.shape[0] == 2 else audio for audio in audio_arrays]

    # Process audio with feature_extractor
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, padding="longest"
    )

    # Assuming your labels are available in the examples
    original_labels = examples["labels"]  # Adjust this based on your actual label column

    # Expand labels to match the desired length
    expanded_labels = []
    for input_values, original_label in zip(inputs["input_values"], original_labels):
        sr = feature_extractor.sampling_rate // 1000
        #target_label_length = (len(input_values) // sr) // 20 #og
        target_label_length = round((len(input_values) / sr) / 20) - 1 
        # Convert original_label to tensor if it's not already
        original_label_tensor = torch.tensor(original_label, dtype=torch.long)

        # If original labels are longer, print a warning and truncate
        if len(original_label_tensor) > target_label_length:
            print(f"Warning: Original label is {len(original_label_tensor) - target_label_length} frame longer than the target. Truncating...")
            original_label_tensor = original_label_tensor[:target_label_length]

        # Pad with zeros if needed -- Thi s needs to be done unbatched
        padding_length = target_label_length - len(original_label_tensor)
        padded_label = torch.cat((original_label_tensor, torch.zeros(padding_length, dtype=torch.long)), dim=0)
        expanded_labels.append(padded_label)

    # One-hot encode the expanded labels
    one_hot_labels = [torch.nn.functional.one_hot(x.long(), num_classes=num_labels) for x in expanded_labels]

    # Add the one-hot encoded labels to the inputs
    inputs['og_labels'] = original_labels
    inputs["labels"] = one_hot_labels
    inputs["audio_path"] = [x["path"] for x in examples["audio"]]
    inputs["words"] = examples["words"]
    inputs["phones"] = examples["phones"]

    return inputs



def compute_metrics(eval_pred):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metrics_d = {metric_name : evaluate.load(metric_name) for metric_name in metric_names}
    num_labels = eval_pred[0].shape[-1]
    logits, labels = eval_pred[0].reshape(-1, num_labels), eval_pred[1].reshape(-1, num_labels)
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)

    metrics = {}
    for metric_name in metric_names:
        metric_value = metrics_d[metric_name].compute(predictions=predictions, references=labels)
        metrics[metric_name] = metric_value[metric_name]

    return metrics


def compute_accuracy(eval_pred):
    num_labels = eval_pred[0].shape[-1]
    metric = evaluate.load("accuracy") # load accuracy metric from datasets
    logits, labels = eval_pred[0].reshape(-1, num_labels), eval_pred[1].reshape(-1, num_labels) # retrieve preds and labels, and reshape them (flatten so that average over all frames in the full dataset)
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_f1(eval_pred):
    num_labels = eval_pred[0].shape[-1]
    metric = evaluate.load("f1") # load accuracy metric from datasets
    logits, labels = eval_pred[0].reshape(-1, num_labels), eval_pred[1].reshape(-1, num_labels) # retrieve preds and labels, and reshape them (flatten so that average over all frames in the full dataset)
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
