# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from emphasis_classifier.utils import infer_utils as emphclass_utils
from transformers import Wav2Vec2ForAudioFrameClassification
from evaluation_pipeline.utils import eval_utils as empheval 
from simalign import SentenceAligner
import random
import whisperx

import os 
import argparse
import sys
import shutil

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def batch_transcribe_whisperx(df, compute_type = "float32", modelname = "large-v2"):
    print("1. Getting transcription and time-alignment")
    
    unique_languages = df['tgt_lang'].unique()
    if len(unique_languages) > 1:
        raise ValueError("Multiple languages detected in the DataFrame. This is not supported. Please split the DataFrame by language and run the evaluation script on each language separately.")
    else:
        lang = unique_languages[0]
    device = "cuda"
    # Load the model once
    model = whisperx.load_model(modelname, device, compute_type=compute_type, asr_options={"suppress_numerals": True}, language=lang)
    
    # Preload align models for all unique languages in the dataframe
    unique_languages = df['tgt_lang'].unique()
    align_model = whisperx.load_align_model(language_code=lang, device=device) 
    
    # Apply the transcribe function to each row in the DataFrame
    df['transcription'], df['word_boundaries'] = zip(*df.progress_apply(lambda row: empheval.decode_whisperx(row, model, align_model), axis=1))

    return df



def get_emphasis(df, emph_model_path):
    print("2. Getting emphasis classification")

    def batch_get_emphasized_words_row(row):
        audio_file = row['tgt_audiopath']
        word_boundaries = row['word_boundaries']

        try:
            pred, emph_boundaries = emphclass_utils.infer_audio(audio_file, emph_model)
            emph_words, tokenized_sentence = empheval.find_emphasized_words_indices(word_boundaries, emph_boundaries, threshold=0.5)
        except:
            raise ValueError("Error in emphasis inference for audio file {}".format(audio_file))

        return emph_words, tokenized_sentence
    
    emph_model =  Wav2Vec2ForAudioFrameClassification.from_pretrained(emph_model_path) 
    df['emph_words'], df['tokenized_sentence'] = zip(*df.progress_apply(lambda row: batch_get_emphasized_words_row(row), axis=1))
    return df   

def align(df, simalign_method = "itermax"):
    print("3. Getting target emphasis indices (by performing translation alignment)")
    translation_aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")
    df['tgt_emphasis'],  df['translation_alignment'] = zip(*df.progress_apply(lambda row: empheval.find_tgt_words_for_row(row, aligner = translation_aligner, simalign_method = simalign_method), axis=1))
    return df




def process_or_load(df, tmp_dir, filename, process_func, *args, **kwargs):
    filepath = os.path.join(tmp_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Using temporary file {filepath} from previous run")
        # Load the DataFrame from a JSON file
        df = pd.read_json(filepath, orient='records', lines=True)
    else:
        # Process the DataFrame using the provided function
        df = process_func(df, *args, **kwargs)
        # Save the processed DataFrame to a JSON file
        df.to_json(filepath, orient='records', lines=True)
    
    return df


def main():
    
    """
    This script is used for evaluation. It can be executed as follows:
    python evaluate.py evaluations/topline_emphasis_small/input.json evaluations/topline_emphasis_small

    The input DataFrame must contain the following columns:

    - 'src_sentence': The source sentence.
    - 'gold_emphasis': The gold standard for emphasis.
    - 'tgt_audiopath': The path to the target audio file.
    - 'id': An identifier to keep track of the data.
    - 'tgt_lang': The language code of the target utterance. This is used to prevent decoding in the wrong language which can cause problems in alignment afterwards (both forced and target).

    """

    parser = argparse.ArgumentParser(description="Runs EmphAssess, the evaluation of emphasis for speech-to-speech models e.g. : python evaluate.py <DF_MODEL> <OUTPUT_DIR>")

    parser.add_argument("df_filename", type=str, help="Path to the input DataFrame file in json format.")
    parser.add_argument("output_dir", help="Directory to save the output DataFrame, results, and potential temporary files.")
    parser.add_argument("emph_model_path",
                        help="Path to the directory containing the emphasis classifier model.")
    parser.add_argument('--use_tmp', action='store_true', 
                        help="Use the latest temporary dataset if it exists in the output_dir/tmp directory. This is useful if the script was previously interrupted.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite the output files if they already exist.")
    parser.add_argument("--simalign_method", choices=["mwmf", "inter", "itermax"], default="itermax", 
                        help="Method to use for simalign. Options are 'mwmf', 'inter', and 'itermax'. Default is 'itermax'.") 
    parser.add_argument("--whispermodel_name", choices=["small", "medium", "large", "large-v2"], default="large-v2", 
                        help="Name of the whisper model to use. Options are 'small', 'medium', 'large', and 'large-v2'. Default is 'large-v2'.") 

    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Check that emph classifier exists
    if not os.path.exists(args.emph_model_path):
        raise ValueError("Emphasis classifier model path {} does not exist".format(args.emph_model_path))

    # if results_df already exists, or results.txt or args.txt, then exit
    if os.path.exists(os.path.join(args.output_dir, "result_df.json")) or os.path.exists(os.path.join(args.output_dir, "results.txt")) or os.path.exists(os.path.join(args.output_dir, "args.txt")):
        if args.overwrite:
            print("Overwriting existing files in {}".format(args.output_dir))
        else:
            print("Output files already exist in {}. Skipping.".format(args.output_dir))
            sys.exit()
    

    
    # Check the file extension and load the DataFrame accordingly
    file_extension = os.path.splitext(args.df_filename)[1].lower()

    if file_extension == '.json':
        df = pd.read_json(args.df_filename, orient='records', lines=True)
    elif file_extension == '.pkl':
        df = pd.read_pickle(args.df_filename)
    else:
        raise ValueError("Unsupported file type. The file must be a .json or .pkl file.")

    tmp_dir = os.path.join(args.output_dir, "tmp")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    df = process_or_load(df, tmp_dir, "df_post1.json", batch_transcribe_whisperx, modelname=args.whispermodel_name)
    df = process_or_load(df, tmp_dir, "df_post2.json", get_emphasis, args.emph_model_path)
    df = process_or_load(df, tmp_dir, "df_post3.json", align, simalign_method=args.simalign_method)


    print("4. Calculating metrics")
    df, averaged_metrics =empheval.get_averaged_metrics(df)
    overall_metrics = empheval.get_overall_metrics(df)
    print("----- \n Overall metrics \n Precision: {0:.2f}, Recall: {1:.2f}, F1 Score: {2:.2f} \n".format(overall_metrics['Precision'], overall_metrics['Recall'], overall_metrics['F1 Score']))
    print("------ \n Averaged metrics \n Precision: {0:.2f}, Recall: {1:.2f}, F1 Score: {2:.2f} \n".format(averaged_metrics['Precision'], averaged_metrics['Recall'], averaged_metrics['F1 Score']))

    df.to_json(os.path.join(args.output_dir, "result_df.json"), orient='records', lines=True)

    
    # Save arguments and results to a file
    with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write("Overall metrics \n Precision: {0:.2f}, Recall: {1:.2f}, F1 Score: {2:.2f} \n".format(overall_metrics['Precision'], overall_metrics['Recall'], overall_metrics['F1 Score']))
        f.write("Averaged metrics \n Precision: {0:.2f}, Recall: {1:.2f}, F1 Score: {2:.2f} \n".format(averaged_metrics['Precision'], averaged_metrics['Recall'], averaged_metrics['F1 Score']))

    print("Results are saved in {}".format(args.output_dir))

    # remove tmp directory if exists
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)



if __name__ == "__main__":
    main()










