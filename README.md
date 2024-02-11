# Easy PyTorch Image Classification

This repository aims to simplify and automate the process of training and evaluating deep learning models for image classification tasks using PyTorch. Whether you are a beginner or an experienced practitioner, The goal is to make your journey in training deep learning models as smooth and efficient as possible.

## Key Features
- **Automated Data Handling:** Automatically handles datasets whether they are compressed, uncompressed but not split, or already split. The repository intelligently processes your data based on its current state, saving you the hassle of manual preparation.
- **Wide Model Support:** Compatible with all models listed on [PyTorch's official model page](https://pytorch.org/vision/stable/models.html), providing you with a vast array of options for your image classification tasks.
- **Comet ML Integration:** Seamlessly integrates with Comet ML for experiment tracking, allowing you to monitor your training processes, compare experiments, and share your results with ease.
- **Simplified Usage:** Designed to be used with minimal parameters for quick setup, yet flexible enough to allow for detailed customization when needed.

## Usage
The main script of this repository is designed to be both powerful and user-friendly, allowing for customization through a variety of parameters, grouped by their functionality:

### Basic Settings
- `-exp`, `--experiment_name`: Specify the experiment name for easy tracking. Default: 'exp'.
- `-dn`, `--dataset_name`: The name of the dataset or the folder name. Use the dataset's name if your data is in "./data/compressed/" (for zipped datasets) or "./data/uncompressed/" (for unzipped datasets). Default: 'ucmerced'.
- `-mn`, `--model_name`: The model you wish to train, should be one of the models available in PyTorch's torchvision models. Default: 'mobilenet_v2'.

### Save Paths
- `-lp`, `--log_path`: Path to save training logs. Default: './results/log/'.
- `-cp`, `--checkpoints_path`: Path to save model checkpoints. Default: './results/checkpoints/'.
- `-ep`, `--evals_path`: Path to save evaluation summaries. Default: './results/evals/'.

### Running Options
- `-tr`, `--train`: Flag to enable model training. Default: False.
- `-ev`, `--eval`: Flag to enable model evaluation. Default: False.
- `-evs`, `--eval_summary`: Flag to generate an evaluation summary. Default: False.

### Hyperparameters
- `-epc`, `--epochs`: Number of training epochs. Default: 100.
- `-bs`, `--batch_size`: Batch size for training/evaluation. Default: 16.
- `-is`, `--image_size`: Image size to resize. Default: 256.
- `-no-aug`, `--no-augmentation`: Flag to disable data augmentation. Default: False
- `-lr`, `--learning_rate`: Learning rate for training. Default: 0.0001.

### Additional Options
- `-nsv`, `--no-save`: Do not save the model and training history. Default: False.
- `-now`, `--no-overwrite`: Prevent overwriting existing models with the same name. Default: False.
- `-npr`, `--no-printing`: Disable printing of progress and details. Default: False.
- `-an`, `--avg_num`: Number of evaluations for averaging in summary. Default: 10.

### Comet ML Integration
- `-uc`, `--use-comet`: Use Comet ML to track and store your training progress. Default: False.
- `-cpn`, `--comet-project-name`: Specify your Comet ML project name. Default: None.
- `-cws`, `--comet-workspace`: Specify your Comet ML workspace. Default: None.
- `-cen`, `--comet-experiment-name`: Specify the Comet ML experiment name. Default: None.
- `-cak`, `--comet-api-key`: Your Comet ML API key. Default: None.


### Minimal Setup Example
To get started with the minimal number of parameters for training a model on a dataset named "ucmerced", with MobileNet v2 as the model:

```
python main.py \
    --dataset_name ucmerced \
    --model_name mobilenet_v2 \
    --train
    --eval
```

This command will automatically handle the dataset preparation (assuming it's placed correctly in one of the "./data/" subdirectories) and start the training process.

### [Comet ML](https://www.comet.com/site/) Integration
To use Comet ML for experiment tracking, simply add the Comet ML flags to your command:

```
python main.py \
    --use-comet \
    --comet-api-key YOUR_API_KEY \
    --comet-project-name YOUR_PROJECT_NAME \
    --comet-workspace YOUR_WORKSPACE \
    --dataset_name ucmerced \
    --model_name mobilenet_v2 \
    --train
    --eval
```

Replace `YOUR_API_KEY`, `YOUR_PROJECT_NAME`, and `YOUR_WORKSPACE` with your Comet ML credentials and desired project/workspace names.

### Dataset Handling
This repository is designed to automate the dataset preparation process:

- For compressed datasets, place your zipped file in "./data/compressed/" and use the same name for -dn or --dataset_name when running the script.
- For uncompressed but not split datasets, place your dataset folder in "./data/uncompressed/" and follow the same naming convention as above.
- If your dataset is already split, place it in "./data/splitted/" and the script will handle it seamlessly.
- The default split ratio is 0.8:0.2 for training and validation sets, respectively. The script takes care of unzipping and splitting the dataset automatically, preparing it for training and evaluation.

Suggestions and Issues
Feel free to drop any suggestions, comments, or report issues you encounter. Your input helps this repository to be more beneficial for everyone in the community.



