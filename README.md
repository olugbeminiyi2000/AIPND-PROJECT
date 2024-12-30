# Flower Species Classifier

This project involves training an image classifier to recognize different species of flowers. Imagine integrating this into a phone app to identify flowers by capturing their images with a camera. The classifier is trained on a dataset containing 102 flower categories and can be exported for practical applications.

## Features

- **Training Data Augmentation**: Enhances training data with random scaling, rotations, mirroring, and cropping using torchvision transforms.
- **Pretrained Network**: Leverages a pretrained model (e.g., VGG16) from `torchvision.models` for feature extraction.
- **Feedforward Classifier**: A new feedforward network is designed to classify images using the features extracted by the pretrained model.
- **Model Saving and Loading**: Supports saving the trained model as a checkpoint, including hyperparameters and `class_to_idx` mapping, and restoring it with a dedicated function.
- **Image Processing**: Includes a function to preprocess images into a format suitable for prediction.
- **Class Prediction**: Predicts the top K probable classes of an input image and provides a visualization with Matplotlib.
- **Command Line Applications**:
  - Train a new model using `train.py`.
  - Predict flower species using `predict.py`.

## Dataset
The project utilizes a dataset of 102 flower categories.

## Rubrics and Implementation

### Part 1 - Development Notebook

#### Package Imports
- All required packages and modules are imported in the first cell of the notebook.

#### Data Preparation
- **Augmentation**: Uses torchvision transforms for scaling, rotations, mirroring, and cropping.
- **Normalization**: Normalizes training, validation, and testing datasets.
- **Data Loading**: Uses `torchvision.datasets.ImageFolder` and `torch.utils.data.DataLoader` for loading datasets.

#### Model
- **Pretrained Network**: Loads a pretrained model (e.g., VGG16) with frozen parameters.
- **Feedforward Classifier**: Implements a custom classifier for flower species recognition.

#### Training
- Trains the classifier parameters while keeping the feature network static.
- Displays validation loss and accuracy during training.

#### Testing
- Measures accuracy on test data.

#### Saving and Loading
- Saves the model checkpoint with hyperparameters and `class_to_idx` dictionary.
- Includes a function to load the checkpoint and rebuild the model.

#### Image Prediction
- Implements `process_image` to convert a PIL image into model input format.
- Includes `predict` to return the top K probable classes for a given image.

#### Sanity Check
- Displays an image with its top 5 most probable classes using Matplotlib.

### Part 2 - Command Line Application

#### Training with `train.py`
- Trains a new model and saves it as a checkpoint.
- Prints training loss, validation loss, and validation accuracy during training.
- Supports selecting from multiple model architectures (e.g., VGG16, ResNet).
- Allows setting hyperparameters such as learning rate, hidden units, and epochs.
- Enables training on GPU if available.

#### Prediction with `predict.py`
- Reads an image and model checkpoint, predicting the most likely class and probability.
- Displays the top K classes and probabilities.
- Loads a JSON file to map class indices to category names.
- Supports GPU-based predictions.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the dataset downloaded and structured appropriately.

## Usage

### Training
Train a new model with `train.py`:
```bash
python train.py --data_dir <data_directory> --save_dir <save_directory> \
--arch <architecture> --learning_rate <learning_rate> --hidden_units <units> \
--epochs <epochs> --gpu
```

### Prediction
Predict the class of a flower image with `predict.py`:
```bash
python predict.py <image_path> <checkpoint_path> --top_k <k> --category_names <json_file> --gpu
```

## Examples

### Training Example
```bash
python train.py --data_dir flowers --save_dir checkpoints --arch vgg16 \
--learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

### Prediction Example
```bash
python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth \
--top_k 5 --category_names cat_to_name.json --gpu
```

## Acknowledgements

This project uses the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and is inspired by practical applications in computer vision and deep learning.
