# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import ast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import librosa
import evaluate
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from IPython.display import Audio, display, FileLink
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (15,3)



#functions to evaluate the model, plot samples etc

def plot_sample(model, ds, sample_index=0, trim_padding=True):
    sample = ds.select([sample_index])
    eval_pred = model.predict(sample)
    metrics = compute_metrics(eval_pred)
    metric = evaluate.load("accuracy") # load accuracy metric from datasets
    logits, labels = eval_pred[0][0], eval_pred[1][0] # retrieve preds and labels, and reshape them (flatten so that average over all frames in the full dataset)
    predicted_labels = np.argmax(logits, axis=-1)
    gold_labels = np.argmax(labels, axis=-1)

    if trim_padding:
        trim_s = len(sample['og_labels'][0])
        predicted_labels = predicted_labels[:trim_s]
        gold_labels = gold_labels[:trim_s]

    predicted_labels_shifted = [x - 2 for x in predicted_labels]

    plt.plot(gold_labels, label='Gold Labels', color='blue')
    plt.plot(predicted_labels_shifted, label='Predicted Labels', color='orange')
    plt.xlabel('Frame Number')
    plt.ylabel('Label Value')
    plt.yticks([-2, -1, 0, 1], ['0 (Predicted)', '1 (Predicted)', '0 (Gold)', '1 (Gold)']) # to label the y-axis
    plt.title('Comparison of Gold and Predicted (argmax) Emphasis Labels')
    plt.legend()
    plt.grid(axis='x', linestyle='--', linewidth=1) # adds thin vertical grid lines
    plt.grid(axis='y', linestyle='--', linewidth=1)   # plt.show()
    plt.show()

def listen_audio(ds, sample_index):
    sample = ds.select([sample_index])
    file_path = sample['audio_path'][0]
    audio_data, sample_rate = librosa.load(file_path)
    audio = Audio(data=audio_data, rate=sample_rate)
    display(audio)
    # Provide a download link for the audio file
    download_link = FileLink(file_path, result_html_prefix="Download the audio file: ")
    display(download_link)

def plot_sample_with_waveform(model, ds, sample_index=0, trim_padding=True, sample_rate=0.02, present_audio=False, word_boundaries = False):
    sample = ds.select([sample_index])
    eval_pred = model.predict(sample)
    metrics = compute_metrics(eval_pred)
    metric = evaluate.load("accuracy") # load accuracy metric from datasets
    logits, labels = eval_pred[0][0], eval_pred[1][0] # retrieve preds and labels, and reshape them (flatten so that average over all frames in the full dataset)
    predicted_labels = np.argmax(logits, axis=-1)
    gold_labels = np.argmax(labels, axis=-1)


    if trim_padding:
        trim_s = len(sample['og_labels'][0])
        predicted_labels = predicted_labels[:trim_s]
        gold_labels = gold_labels[:trim_s]

    # Load your audio file
    file_path = sample['audio_path'][0]
    audio_data, sample_rate = librosa.load(file_path)

    # Plot the figure
    plt.figure(figsize=(15, 6))

    # Determine the frame length for overlay
    frame_length = 0.02

 # Function to overlay word boundaries
    def overlay_word_boundaries(word_b):
        for word, start_time, end_time in word_b:
            plt.axvline(x=start_time, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x=end_time, color='k', linestyle=':', alpha=0.5)
            middle_time = (start_time + end_time) / 2
            plt.text(middle_time, 0.5, word, color='k', ha='center')


    # Overlay gold labels
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate, alpha=.7)
    for i, value in enumerate(gold_labels):
        if value == 1:
            plt.axvspan(i * frame_length, (i + 1) * frame_length, color='g', alpha=0.2)
    if word_boundaries:
        word_b = ast.literal_eval(sample['words'][0])
        overlay_word_boundaries(word_b)
    plt.title('Gold Emphasis Labels')

    # Overlay predicted labels
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio_data, sr=sample_rate, alpha=.7)
    for i, value in enumerate(predicted_labels):
        if value == 1:
            plt.axvspan(i * frame_length, (i + 1) * frame_length, color='r', alpha=0.2)
    if word_boundaries:
        overlay_word_boundaries(word_b)
    plt.title('Predicted Emphasis Labels')

    plt.suptitle("Gold (top) vs Predicted (bottom) emphasis labels", fontsize=14)
    plt.tight_layout()
    plt.show()

    if present_audio:
        audio = listen_audio(ds, sample_index)

def plot_sample_with_waveform_proba(model, ds, sample_index=0, trim_padding=True, sample_rate=0.02, present_audio=False, word_boundaries = False):
    sample = ds.select([sample_index])
    eval_pred = model.predict(sample)
    logits, labels = eval_pred[0][0], eval_pred[1][0] # retrieve preds and labels, and reshape them (flatten so that average over all frames in the full dataset)
    prob_scores = np.exp(logits[:, 1]) / (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
    
    predicted_labels = np.argmax(logits, axis=-1)
    gold_labels = np.argmax(labels, axis=-1)


    if trim_padding:
        trim_s = len(sample['og_labels'][0])
        predicted_labels = predicted_labels[:trim_s]
        gold_labels = gold_labels[:trim_s]
        prob_scores = prob_scores[:trim_s]


    # Load your audio file
    file_path = sample['audio_path'][0]
    audio_data, sample_rate = librosa.load(file_path)

    # Plot the figure
    plt.figure(figsize=(15, 6))

    # Determine the frame length for overlay
    frame_length = 0.02

    audio_data = audio_data / np.max(np.abs(audio_data))

    # max_amplitude = max(abs(audio_data))


 # Function to overlay word boundaries
    def overlay_word_boundaries(word_b):
        for word, start_time, end_time in word_b:
            plt.axvline(x=start_time, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x=end_time, color='k', linestyle=':', alpha=0.5)
            middle_time = (start_time + end_time) / 2
            # plt.text(middle_time, 0.5, word, color='k', ha='center')
            plt.text(middle_time, 1.2, word, color='k', ha='center')



    # Overlay gold labels
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=sample_rate, alpha=.7)
    # plt.plot(np.arange(len(prob_scores)) * frame_length, prob_scores, label='Probability of 1', color='b')
    for i, value in enumerate(gold_labels):
        if value == 1:
            plt.axvspan(i * frame_length, (i + 1) * frame_length, color='g', alpha=0.2)
    if word_boundaries:
        word_b = ast.literal_eval(sample['words'][0])
        overlay_word_boundaries(word_b)
    plt.legend()

    # Overlay predicted labels
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio_data, sr=sample_rate, alpha=.7)
    plt.plot(np.arange(len(prob_scores)) * frame_length, prob_scores, label='Probability of emphasis', color='r')
    for i, value in enumerate(predicted_labels):
        if value == 1:
            plt.axvspan(i * frame_length, (i + 1) * frame_length, color='r', alpha=0.2)
    if word_boundaries:
        overlay_word_boundaries(word_b)
    plt.legend()
    #plt.title('Predicted Emphasis Labels')
    # plt.ylim([-max_amplitude, max_amplitude + 0.5])  # The 0.5 ensures the probability line is visible above the waveform.
    # plt.ylim([-1, 1.5])  # This ensures the probability line is visible above the waveform.


    plt.suptitle("Gold (top) vs Predicted (bottom) emphasis labels", fontsize=14)
    plt.tight_layout()
    plt.show()
    # print("file_path:", file_path)

    if present_audio:
        audio = listen_audio(ds, sample_index)



def evaluate_word_level(model, ds, trim_padding=True):
    ## to do : add padding trim
    eval_pred = model.predict(ds) # get predictions

    return metrics

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


def compute_word_level_metrics(eval_pred, word_boundaries, threshold=0.5):
    # Extract predictions and reshape
    num_labels = eval_pred[0].shape[-1]
    logits, labels = eval_pred[0], eval_pred[1]

    # Initialize word-level predictions and labels
    word_predictions = []
    word_labels = []

    # Iterate through word boundaries and compute word-level predictions and labels
    for i, boundaries in enumerate(word_boundaries):
        frame_predictions = np.argmax(logits[i], axis=-1)
        frame_labels = np.argmax(labels[i], axis=-1)
        boundaries = ast.literal_eval(boundaries)

        for word, start_time, end_time in boundaries:
            start_frame = int(start_time / 0.02) # Assuming 0.02s per frame
            end_frame = int(end_time / 0.02)
            
            # Average the frame-level predictions and labels for the current word
            word_pred = np.mean(frame_predictions[start_frame:end_frame])
            word_label = np.mean(frame_labels[start_frame:end_frame])

            # Apply threshold
            word_predictions.append(int(word_pred >= threshold))
            word_labels.append(int(word_label >= threshold))



    # Compute Metrics
    metrics = {
        "accuracy": accuracy_score(word_labels, word_predictions),
        "precision": precision_score(word_labels, word_predictions),
        "recall": recall_score(word_labels, word_predictions),
        "f1": f1_score(word_labels, word_predictions),
    }

    return metrics, classification_report(word_labels, word_predictions)





def compute_metrics_for_df(eval_pred, indices, dataset, threshold=0.5):
    metrics = [("accuracy", accuracy_score), ("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]
    
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    labels = np.argmax(labels, axis=-1)
    predictions = np.argmax(logits, axis=-1)

    metrics_per_utterance = []
    word_level_metrics_per_utterance = []

    for idx, i in enumerate(indices):
        single_pred = list(predictions[idx])
        single_label = list(labels[idx])
        
        metrics_dict = {}
        for metric_name, metric_function in metrics:
            metric_value = metric_function(single_label, single_pred)
            metrics_dict[metric_name] = metric_value
        
        metrics_per_utterance.append(metrics_dict)

        # Word-level metrics computation
        word_predictions = []
        word_labels = []

        boundaries = ast.literal_eval(dataset[i]["words"])

        for word, start_time, end_time in boundaries:
            start_frame = int(start_time / 0.02)  # Assuming 0.02s per frame
            end_frame = int(end_time / 0.02)
            
            # Average the frame-level predictions and labels for the current word
            word_pred = np.mean(single_pred[start_frame:end_frame])
            word_label = np.mean(single_label[start_frame:end_frame])

            # Apply threshold for majority voting
            word_predictions.append(int(word_pred >= threshold))
            word_labels.append(int(word_label >= threshold))

        word_level_metrics = {
            "word_accuracy": accuracy_score(word_labels, word_predictions),
            "word_precision": precision_score(word_labels, word_predictions),
            "word_recall": recall_score(word_labels, word_predictions),
            "word_f1": f1_score(word_labels, word_predictions),
        }

        word_level_metrics_per_utterance.append(word_level_metrics)

    return predictions, labels, metrics_per_utterance, word_level_metrics_per_utterance, indices

def create_df_withresults(model , ds, selected_indices = None ):
    if selected_indices is None:
        selected_indices = list(range(len(ds)))

    eval_pred = model.predict(ds.select(selected_indices))
    predictions, labels, frame_metrics, word_metrics, returned_indices = compute_metrics_for_df(eval_pred, selected_indices, ds)


    audio_paths = [ds[index]['audio_path'] for index in returned_indices]
    speakers = [ds[index]['speaker'] for index in returned_indices]
    styles = [ds[index]['style'] for index in returned_indices]
    words = [ds[index]['words'] for index in returned_indices]
    phones = [ds[index]['phones'] for index in returned_indices]

    df = pd.DataFrame({
        'Index': returned_indices,
        'AudioPath': audio_paths,
        'name': [os.path.splitext(os.path.basename(filename))[0] for filename in audio_paths],
        'Speaker': speakers,
        'Style': styles,
        'Words': words,
        'Phones': phones,
        'Predictions': list(predictions),
        'Labels': list(labels),
        'frame_accuracy': [m['accuracy'] for m in frame_metrics],
        'frame_precision': [m['precision'] for m in frame_metrics],
        'frame_recall': [m['recall'] for m in frame_metrics],
        'frame_f1': [m['f1'] for m in frame_metrics],
        'word_accuracy': [m['word_accuracy'] for m in word_metrics],
        'word_precision': [m['word_precision'] for m in word_metrics],
        'word_recall': [m['word_recall'] for m in word_metrics],
        'word_f1': [m['word_f1'] for m in word_metrics],
    })

    return df