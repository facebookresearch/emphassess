# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.emphasis_classifier.utils import dataset_utils
from transformers import AutoFeatureExtractor, Wav2Vec2ForAudioFrameClassification
from transformers import TrainingArguments, Trainer

from datasets import load_dataset, Audio
import torch
import os
from datetime import datetime


def prepare_dataset(data_dir, fe_model = "facebook/wav2vec2-xls-r-1b", num_labels = 2):
    dataset = load_dataset("audiofolder", data_dir=data_dir)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) # 16khz
    sampling_rate = dataset["train"].features["audio"].sampling_rate

    # Apply the function over the dataset
    dataset = dataset.map(dataset_utils.add_label, remove_columns=["emphasis"], batched=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(fe_model)
    encoded_ds = dataset.map(dataset_utils.preprocess_function, remove_columns="audio", batched=True, batch_size=max([len(dataset[k]) for k in dataset.keys()]), fn_kwargs={'feature_extractor': feature_extractor, 'num_labels': num_labels})
    return encoded_ds

def login_hf(token):
    from huggingface_hub import login
    with open(token, 'r') as f:
        hf_token = f.read().replace('\n', '')
    login(token=hf_token)

def finetune_model(pretrained_model, train_ds, validation_ds, output_dir, n_epochs=10, compute_metric = dataset_utils.compute_metrics, logging_steps = 10, lr = 3e-5, eval_batch_size=32,  
                   warmup_ratio=0.309, train_batch_size=8):
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(pretrained_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model)


    model.freeze_feature_encoder()





    run_name = output_dir.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=output_dir,evaluation_strategy="epoch", save_strategy="epoch", learning_rate=lr,
        per_device_train_batch_size=8, gradient_accumulation_steps=3, per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=n_epochs, warmup_ratio=warmup_ratio, logging_steps=logging_steps, load_best_model_at_end=True,
        push_to_hub=False, run_name = run_name,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,args=training_args,
        train_dataset=train_ds, eval_dataset=validation_ds,
        tokenizer=feature_extractor, compute_metrics=compute_metric

    )

    trainer.train()

    return trainer


def load_finetuned_model(model_dir):
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(model_dir)
    return model


if __name__ == "__main__":

    from huggingface_hub import login
    from datasets import disable_caching
    disable_caching() #When you disable caching, HF Datasets will no longer reload cached files when applying transforms to datasets. Any transform you apply on your dataset will be need to be reapplied.


    # login to huggingface hub
    with open('hf_token.txt', 'r') as f:
        hf_token = f.read().replace('\n', '')
    login(token=hf_token)


    import argparse

    parser = argparse.ArgumentParser(description="Your script description here")

    parser.add_argument("checkpoint_dir", type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("datadir", type=str, help="Path to the directory containing the data in hf format with labels in metadata")
    parser.add_argument("--run_name", default=None, type=str, help="Custom run name. If provided, it will override the automatically generated run name.")
    parser.add_argument("--suffix", default=None, type=str, help="Suffix to add to run name. If provided, it will be added to the automatically generated run name. Doesn't work with args.run_name.")

    parser.add_argument("--pretrained_model", default = "facebook/wav2vec2-xls-r-1b")
    parser.add_argument("--n_epochs", default = 15, type=int)
    parser.add_argument("--learning_rate", default = 0.0000797, type=float)
    parser.add_argument("--eval_batch_size", default = 16, type=int)
    parser.add_argument("--train_batch_size", default = 8, type=int)
    parser.add_argument("--warmup_ratio", default = 0.12488, type=float)


    parser.add_argument("--hf_token", default = 'hf_token.txt') #need to setup to be able to use huggingface hub
    

    parser.add_argument("--subset_val_data", action="store_true",
            help="Subset to 100 utterances in val data if True")
    parser.add_argument("--subset_train_data", action="store_true",
            help="Subset to 400 utterances in train data if True")
    parser.add_argument("--test_run", action="store_true",
            help="test_run")







    args = parser.parse_args()

    login_hf(token=args.hf_token)
    ds = prepare_dataset(data_dir=args.datadir, fe_model=args.pretrained_model) # Carefiul because path is wrng in the metadata. Plus possibly need full path not relative. 
    #todo : have al anready prepared dataset saved somewhere. 

    # Get current date and time

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pretrained_model_name = args.pretrained_model.split("/")[-1]


    if args.run_name is None:
        run_name = f"{pretrained_model_name}_lr-{args.learning_rate}_epochs-{args.n_epochs}_warmup-{args.warmup_ratio}_batch-{args.train_batch_size}_evalbatch-{args.eval_batch_size}"
        run_name += f"_time-{current_time}"


        if args.subset_val_data:
            run_name += "_subsetval"
        if args.subset_train_data:
            run_name += "_subsettrain"
        if args.test_run:
            run_name += "_testrun"
        if args.suffix:
            run_name += f"_{args.suffix}"
    else:
        run_name = args.run_name


    if args.test_run:
        train_ds = ds["train"].shuffle(seed=42).select(range(6))
        validation_ds = ds["validation"].shuffle(seed=42).select(range(3))
    else:
        if args.subset_val_data:
            validation_ds = ds["validation"].shuffle(seed=42).select(range(100))
        else:
            validation_ds = ds["validation"]
        if args.subset_train_data:
            train_ds = ds["train"].shuffle(seed=42).select(range(400))
        else:
            train_ds = ds["train"]


    print("Training model with following command line arguments:")
    print("Run name: {}".format(run_name))
    print(args)


    output_dir = os.path.join(args.checkpoint_dir, run_name)
    print("OUTPUT DIR: {}".format(output_dir))
    trainer = finetune_model(args.pretrained_model, train_ds, validation_ds, output_dir, n_epochs=args.n_epochs,
                              lr = args.learning_rate, eval_batch_size=args.eval_batch_size,  
                              train_batch_size=args.train_batch_size, warmup_ratio=args.warmup_ratio, 
                              logging_steps = 10)


    # Compute metrics for a sample of the validation set
    print("Computing metrics")
    eval_pred = trainer.predict(validation_ds)
    metrics = dataset_utils.compute_metrics(eval_pred)
    for k,v in metrics.items():
        print("{}: {}%".format(k, round(v*100, 2)))