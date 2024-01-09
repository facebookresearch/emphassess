# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from emphasis_classifier.utils import infer_utils
import os 
import matplotlib.pyplot as plt
import argparse


import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def main():

    """
    Main function to perform emphasis classification inference on an audio file and save the results.
    Writes a text file with the emphasized intervals and saves a plot of the audio file with the emphasized intervals highlighted (in seconds).

    Parameters:
    
    audio_file (str): Path to the audio file to infer.
    model_path (str): Path to the pretrained model.
    save_dir (str): Directory where the output will be saved.
    save_plot (bool, optional): Whether to save the output plot. Defaults to False.

    Returns:
    None
    """

    parser = argparse.ArgumentParser(description='Infer audio file with model.')
    parser.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser.add_argument('--save_dir', type=str, default = "output_classification",help='Path to save the output plot and labels for classifdirectory. Will be saved with the name of the audio file.')
    parser.add_argument('--model_path', type=str, default = "src/emphasis_classifier/checkpoints/en", help='Path to the model directory.')
    parser.add_argument('--save_plot', default=False, type=bool, help='Display the plot.')
    args = parser.parse_args()

    audio_file = args.audio_file
    model_path = args.model_path
    save_dir = args.save_dir
    save_plot = args.save_plot



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Load the model
    
    model = infer_utils.Wav2Vec2ForAudioFrameClassification.from_pretrained(model_path)

    # Retrieve emphasis boundaries and predictions
    pred, emph_boundaries = infer_utils.infer_audio(audio_file, model)

    print("Emphasis boundaries (in seconds): ", emph_boundaries)


    # Get the base name of the audio file
    audio_basename = os.path.basename(audio_file)

    output_file = os.path.splitext(audio_basename)[0] + '.txt'

    with open(os.path.join(save_dir, output_file), 'w') as f:
        for start, end in emph_boundaries:
            # Write the interval to the file formatted to two decimal places
            f.write(f"{start:.2f}-{end:.2f}\n")

    print(f"Emphasized intervals saved to {os.path.join(save_dir, output_file)}")


    if save_plot:
        # Plot output
        fig = infer_utils.plot_output(pred, audio_file)
        output_plot = os.path.splitext(audio_basename)[0] + '.png'
        
        fig.savefig(os.path.join(save_dir, output_plot))

    print("Output plot saved to {}".format(save_dir))

    

if __name__ == "__main__":


    main()