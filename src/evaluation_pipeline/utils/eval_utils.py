# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import pipeline
import re
import warnings
import random 
import pandas as pd 
import whisperx


def decode_whisperx(row, model, align_model):

    device = "cuda"
    batch_size = 16  # reduce if low on GPU mem

    # 1. Transcribe with original whisper (batched)
    audio = whisperx.load_audio(row['tgt_audiopath'])

    # code with buffer below is not to have this additional print tatements rgearding tokens
    import io
    from contextlib import redirect_stdout
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        result = model.transcribe(audio, batch_size=batch_size, language = row['tgt_lang'])

    # Reset buffer position
    buffer.seek(0)

    # Read from buffer line by line
    for line in buffer:
        if "Suppressing numeral and symbol tokens" not in line:
            print(line.strip())  # Use strip() to remove any leading/trailing whitespace or newlines


    # Retrieve the align model for this language from the dictionary
    model_a, metadata = align_model
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    try:

        text = result['segments'][0]['text']
        word_boundaries = [(x["word"], x["start"], x["end"]) for x in result["segments"][0]["words"]]
    except: # sometimes exception when no word boundaries found, mainly because of incorrect transcription (rare instances)
        text = None
        word_boundaries = []

    return text, word_boundaries

def merge_characters(char_time_tuples): 
    merged_tuples = []
    current_word = ""
    start_time = None
    end_time = None

    for char, start, end in char_time_tuples:
        if char == "no_speech":
            if current_word:  # If a word has been accumulated
                merged_tuples.append((current_word, start_time, end_time))
            current_word = ""
            start_time = None
        else:
            if start_time is None:
                start_time = start
            end_time = end
            current_word += char

    if current_word:  # For the last word
        merged_tuples.append((current_word, start_time, end_time))

    return merged_tuples


def decode_whisper(row, model, force_language = True):
    audio_file = row['tgt_audiopath']
    if force_language:
        result = model.transcribe(audio_file, language = row['tgt_lang'])
    else:   
        result = model.transcribe(audio_file)
    text = result["text"]
    return text




def find_tgt_words_for_row(row, aligner, model=None, tokenizer=None, simalign_method="itermax"):
    src_sentence_tokenized = row['src_sentence']
    tgt_sentence_tokenized = row['tokenized_sentence']
    gold_emph_words = row['gold_emphasis']

    try:
        alignments = aligner.get_word_aligns(src_sentence_tokenized, tgt_sentence_tokenized)
        tgt_emphasis = find_target_emphasis_indices(src_sentence_tokenized, tgt_sentence_tokenized, alignments, gold_emph_words, method=simalign_method)
    except:
        warnings.warn("Error in finding target emphasis words for filename: {}, giving it scores of 0. ".format(row["tgt_audiopath"]))
        tgt_emphasis = []
        alignments = []

    return tgt_emphasis, alignments




def find_emphasized_words_indices(word_boundaries, emph_boundaries, threshold=0.5):
    """
    Finds which words are emphasized based on their time boundaries and the boundaries of emphasized segments.
    
    Parameters:
    - word_boundaries (list of tuple): A list of tuples, each containing a word and its start and end times.
    - emph_boundaries (list of tuple): A list of tuples, each containing the start and end times of an emphasized segment.
    - threshold (float): The minimum fraction of a word's duration that needs to be emphasized for the word to be considered emphasized.
    
    Returns:
    - list: A list of tuples, each containing an emphasized word and its index in the word_boundaries list.
    - list: The tokenized sentence.
    """
    
    emphasized_words = []  # List to store the emphasized words
    tokenized_sentence = [word for word, _, _ in word_boundaries]  # Create the tokenized sentence
    
    # Loop through each word and its boundaries
    for index, (word, word_start, word_end) in enumerate(word_boundaries):
        word_duration = word_end - word_start  # Calculate the duration of the word
        overlap_duration = 0.0  # Initialize the duration of overlap with emphasized segments
        
        # Loop through each emphasized segment and its boundaries
        for emph_start, emph_end in emph_boundaries:
            overlap_start = max(word_start, emph_start)  # Calculate the start of the overlapping segment
            overlap_end = min(word_end, emph_end)  # Calculate the end of the overlapping segment
            
            # If there is an overlap, add its duration to overlap_duration
            if overlap_start < overlap_end:
                overlap_duration += (overlap_end - overlap_start)
        
        # Check if the word is emphasized based on the threshold
        if overlap_duration > word_duration * threshold:
            # emphasized_words.append((word, index))
            emphasized_words.append(index)
    
    return emphasized_words, tokenized_sentence



def clean_word(word): 
    # Remove punctuation at the start of the word
    word = re.sub(r'^[^a-zA-Z0-9]+', '', word)
    
    # Remove punctuation at the end of the word
    word = re.sub(r'[^a-zA-Z0-9]+$', '', word)

    return word


def find_target_emphasis_indices(src_sentence_tokenized, tgt_sentence_tokenized, alignments, src_emphasis_indices, method='itermax'):
    """
    Given a source sentence, target sentence, their word alignments, and indices to be emphasized in the source,
    this function returns the corresponding indices to be emphasized in the target along with the tokenized target sentence.

    :param src_sentence_tokenized: Tokenized source sentence
    :param tgt_sentence_tokenized: Tokenized target sentence
    :param alignments: A dictionary of word alignments (could be from different methods)
    :param src_emphasis_indices: A list of indices to be emphasized in the source sentence
    :param method: The alignment method to use (default is 'itermax')
    :return: A tuple containing the tokenized target sentence and a list of indices to be emphasized in the target sentence
    """

    tgt_emphasis_indices = []

    if method not in alignments:
        return f"Method '{method}' not found in alignments."

    for src_index, tgt_index in alignments[method]:
        if src_index in src_emphasis_indices:
            tgt_emphasis_indices.append(tgt_index)

    # Remove duplicates and sort
    tgt_emphasis_indices = sorted(set(tgt_emphasis_indices))

    return tgt_emphasis_indices



def get_overall_metrics(df):
    # Get metrics over the whole dataset all at once

    def aggregate_counts(row):
        src_emphasis_indices = set(row['emph_words'])
        tgt_emphasis_indices = set(row['tgt_emphasis'])
        
        TP = len(src_emphasis_indices.intersection(tgt_emphasis_indices))
        FP = len(tgt_emphasis_indices.difference(src_emphasis_indices))
        FN = len(src_emphasis_indices.difference(tgt_emphasis_indices))
        
        return pd.Series([TP, FP, FN], index=['TP', 'FP', 'FN'])

    # Aggregate counts for the entire dataframe
    df_counts = df.apply(aggregate_counts, axis=1).sum()

    # Compute overall metrics
    TP = df_counts['TP']
    FP = df_counts['FP']
    FN = df_counts['FN']

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    overall_metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1_score}

    return overall_metrics

def get_averaged_metrics(df):
    # Get first metrics rowwise then average over the whole dataset
    def calculate_metrics_for_row(row):
        src_emphasis_indices = row['emph_words']  # Assuming you have this column in your dataframe
        tgt_emphasis_indices = row['tgt_emphasis']  # Assuming you have this column in your dataframe
        return pd.Series(calculate_emphasis_metrics(src_emphasis_indices, tgt_emphasis_indices))


    df[['Precision', 'Recall', 'F1 Score']] = df.apply(calculate_metrics_for_row, axis=1)
    overall_metrics = df[['Precision', 'Recall', 'F1 Score']].mean()
    return df, overall_metrics


def calculate_emphasis_metrics(src_emphasis_indices, tgt_emphasis_indices):
    """
    Calculate Precision, Recall, F1 Score for emphasis translation.
    Precision: Of the indices emphasized in the target sentence, how many were actually supposed to be emphasized (i.e., are emphasized in the source sentence)?
    Recall: Of the indices that were supposed to be emphasized (i.e., are emphasized in the source sentence), how many were actually emphasized in the target sentence?
    F1 Score: The harmonic mean of Precision and Recall, providing a balance between the two.

    :param src_emphasis_indices: List of emphasized word indices in the source sentence (gold emphasis).
    :type src_emphasis_indices: list[int]
    :param tgt_emphasis_indices: List of emphasized word indices in the target sentence (actual emphasis).
    :type tgt_emphasis_indices: list[int]
    :return: Dictionary containing Precision, Recall, and F1 Score.
    :rtype: dict
    """
    
    true_positive = len(set(src_emphasis_indices).intersection(set(tgt_emphasis_indices)))
    
    if len(tgt_emphasis_indices) == 0:
        precision = 0
    else:
        precision = true_positive / len(tgt_emphasis_indices)
        
    if len(src_emphasis_indices) == 0:
        recall = 0
    else:
        recall = true_positive / len(src_emphasis_indices)
        
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {"Precision": precision, "Recall": recall, "F1 Score": f1_score}
