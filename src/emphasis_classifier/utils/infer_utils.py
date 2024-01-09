# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import Wav2Vec2ForAudioFrameClassification
import numpy as np
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt


def infer_audio(audio_file, model):

    # Load and preprocess the audio
    input_tensor, sample_rate = preprocess_single_audio(audio_file)  

    # Run the forward pass
    with torch.no_grad():
        outputs = model(input_tensor)


    res = np.argmax(outputs["logits"], axis=-1).flatten()
    emph_boundaries = find_emphasis_boundaries(res, sample_rate)
    return res, emph_boundaries

def preprocess_single_audio(audio_file_path, sampling_rate=16000):
    # Load the audio file
    waveform, sr = torchaudio.load(audio_file_path)

    # Convert to mono if stereo
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if the sampling rate is different
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        waveform = resampler(waveform)

    return waveform, sampling_rate


def find_emphasis_boundaries(output_tensor, sample_rate, frame_size=20):
    frame_length = int(sample_rate *  frame_size/1000)  # Assuming frames are 20ms long

    # Initialize variables
    start_time = None
    end_time = None
    segments = []

    # Loop through the tensor to find start and end times
    for i, value in enumerate(output_tensor):
        if value == 1:
            if start_time is None:
                start_time = i * frame_length / sample_rate  # Convert to seconds
            end_time = (i + 1) * frame_length / sample_rate  # Convert to seconds
        else:
            if start_time is not None:
                segments.append((start_time, end_time))
                start_time = None
                end_time = None

    # Handle case where the last segment goes up to the last frame
    if start_time is not None:
        segments.append((start_time, end_time))

    return segments




# def plot_output(output_tensor, audio_file):
#     # Load the audio file
#     audio_data, sample_rate = librosa.load(audio_file, sr=None)


#     # Parameters
#     frame_length = int(sample_rate * 0.02)  # Assuming frames are 20ms long
#     num_frames = len(output_tensor)

#     # Plotting
#     plt.figure(figsize=(20, 6))

#     # Plot the waveform using librosa
#     librosa.display.waveshow(audio_data, sr=sample_rate, alpha=.7, color = "blue")

#     # Initialize a flag for the label
#     label_set = False
#     # Highlight the frames where the tensor is 1
#     for i in range(num_frames):
#         if output_tensor[i] == 1:
#             start = i * frame_length / sample_rate  # Convert to seconds
#             end = (i + 1) * frame_length / sample_rate  # Convert to seconds
#             label = 'Emphasized' if not label_set else ""
#             plt.axvspan(start, end, color='red', alpha=0.5, label=label)
#             label_set = True  # Set the flag to True after the first label

#     plt.suptitle('Waveform with Emphasized Frames')
#     plt.title('Audio file: {}'.format(audio_file.split('/')[-1]))
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.show()

import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_output(output_tensor, audio_file):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Parameters
    frame_length = int(sample_rate * 0.02)  # Assuming frames are 20ms long
    num_frames = len(output_tensor)

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot the waveform using librosa
    librosa.display.waveshow(audio_data, sr=sample_rate, alpha=.7, color = "blue", ax=ax)

    # Initialize a flag for the label
    label_set = False
    # Highlight the frames where the tensor is 1
    for i in range(num_frames):
        if output_tensor[i] == 1:
            start = i * frame_length / sample_rate  # Convert to seconds
            end = (i + 1) * frame_length / sample_rate  # Convert to seconds
            label = 'Emphasized' if not label_set else ""
            ax.axvspan(start, end, color='red', alpha=0.5, label=label)
            label_set = True  # Set the flag to True after the first label

    ax.set_title('Waveform with Emphasized Frames\nAudio file: {}'.format(audio_file.split('/')[-1]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()

    return fig



def get_emphasized_intervals(output_tensor, audio_file):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    
    # Parameters
    frame_length = int(sample_rate * 0.02)  # Assuming frames are 20ms long
    num_frames = len(output_tensor)
    
    # List to hold the start and end times of emphasized intervals
    emphasized_intervals = []

    # Detect and store the intervals
    for i in range(num_frames):
        if output_tensor[i] == 1:
            start_time = i * frame_length / sample_rate  # Convert to seconds
            end_time = (i + 1) * frame_length / sample_rate  # Convert to seconds
            emphasized_intervals.append((start_time, end_time))
    
    return emphasized_intervals

