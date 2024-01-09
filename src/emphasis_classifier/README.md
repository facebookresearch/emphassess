# Emphasis Classification Model

This repository contains the code necessary for training and running inference with our emphasis assessment model, designed primarily for emphasis detection in spoken language.

Please not, the scripts contained in this drectory are only here for modifications, but we cnanot endsre they can be run as they are, except from the utils functions needed for EmphAssess evaluation.

## Model Checkpoint

The model checkpoint is available for download:
- [Model Checkpoint](https://dl.fbaipublicfiles.com/speech_expressivity_evaluation/EmphAssess/EmphaClass/EmphaClass-en.tar.gz)

It is also available in the repo at `checkpoints/en`

## Training

The training script can be found at `train_model.py`. Please note that this script is provided as a baseline and reauires modifications to suit your specific use case.

### Hardware Requirements
- A GPU is recommended for training the model efficiently.

## Data Preparation

Data must be force-aligned to generate emphasis labels. Labels correspond to 20ms frames, marked as 0 (non-emphasized) or 1 (emphasized). Ensure your data is formatted as a Hugging Face dataset with the following partitions: train, validation, and test.


### Training Script
- Located at: `./scripts/train_model.py`

## Dependencies

Install the required Python packages before starting:

```bash
pip install datasets wandb
```


# Inference

Inference can be performed using infer_audio.py. Please note that the script is intended as a starting point and might need adjustments for your use case.

Execute the inference script snipper as follows:

`python infer_audio.py --audio_path /path/to/your/audio/file`



