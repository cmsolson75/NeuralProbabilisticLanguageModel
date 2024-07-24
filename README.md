# Neural Probabilistic Language Model

*Implementation of the 2003 paper "A Neural Probabilistic Language Model"  by Yoshua Bengio.*
*Inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" lecture series.*

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Name Generation App](#name-generation-app)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [File Structure](#file-structure)

## Features

### Inference App: `app.py`
- Generate names given a short prompt.

### Custom Implementation
- **Custom Optimizer**: SGD + Momentum + Weight Decay
  - Weight Decay on update
  - Momentum tracking
  - Custom step function for SGD
- **Custom Trainer**
  - Automatic device discovery
  - Linear learning rate decay
  - Fit function with `tqdm` for progress bar
  - Evaluation for model tuning using the perplexity metric `torch.exp(F.cross_entropy(logits, targets))`, as described in the paper.
- **Model Implementation**
  - Custom serialization
  - Custom model load
  - Custom linear layer
  - Weight initialization using custom Xavier implementations (uniform & normal)
- **Data Loaders**
  - Parse text dataset
  - Random split into train, test, and dev sets.
- **Minimal Use of PyTorch**
  - Only uses PyTorch for CrossEntropyLoss and autograd for backpropagation.
  - Utilizes `torch.Generator()` for reproducibility.
- **Custom Hyperparameter Tuning**
  - Uses Random Sampling over the hyperparameter space.
  - Test NLL: 1.9882903099060059
  - Test Perplexity: 7.303037166595459
  - Best parameters:
    ```json
    {
      "lr_start": 0.1,
      "lr_end": 0.001,
      "h_size": 328,
      "context_size": 6,
      "emb_dim": 24,
      "weight_decay": 0.00022564512422341903,
      "momentum": 0.9,
      "batch_size": 512
    }
    ```

## Setup

### Requirements
Ensure you have the correct python version installed:
- Python >= 3.10

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration
Configurations are stored in the `configs` directory. Ensure that your configuration files are correctly set up before running the training or inference scripts.

### Model Training
To train the model, use the `train.py` script with the appropriate configuration file.
```bash
python train.py --config configs/config.json
```

### Name Generation App
Run the name generation application using:
```bash
python app.py --prompt "Prompt Here" --num_names 5 --temperature 0.7
```
To modify the model used change path in `configs/app_config.json`

### Hyperparameter Tuning
Perform hyperparameter tuning using the `hyperparameter_search.py` script.
```bash
python hyperparameter_search.py
```

## Configuration

### Configuration File Structure
A typical configuration file (`configs/config.json`) looks like:
```json
{
    "runName": "gpt_but_2003",
    "dataPath": "data/names.txt",
    "epochs": 1000,
    "batchSize": 512,
    "learningRateDecay":[0.1, 0.001],
    "vocab": 27,
    "hiddenSize": 350,
    "embeddingSize": 12,
    "context": 6,
    "weightInitialization": "normal",
    "weightDecay": 0.0001,
    "momentum": 0.9,
    "generatorSeed": 42
}
```
