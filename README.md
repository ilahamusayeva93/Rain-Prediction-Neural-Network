# Rain Prediction Neural Network Case Study

## Overview

In this case study, we aim to build a neural network using PyTorch to predict whether it will rain tomorrow based on real weather information. The dataset used for training and evaluation includes various weather features, and the target variable is binary, indicating whether it will rain or not.

## Dataset

The weather dataset is loaded from the "weather.csv" file. The following preprocessing steps are performed:

- Handling missing values using backfill and dropping remaining NaNs.
- Encoding categorical variables using LabelEncoder.
- Extracting additional features from the "Date" column (Year, Month, Day).
- Splitting the dataset into input features (X) and the target variable (y).
- Performing oversampling using ADASYN to handle class imbalance.
- Standardizing numerical features using StandardScaler.

## Neural Network Architecture

The neural network architecture consists of the following layers:

1. Input Layer: Number of neurons equal to the number of input features.
2. Fully Connected (Linear) Layer with Batch Normalization and ReLU Activation.
3. Dropout Layer to prevent overfitting.
4. Output Layer with a single neuron and no activation function.

## Training

The neural network is trained with the following hyperparameters:

- Learning Rate: 0.001
- Weight Decay: 0.001
- Number of Epochs: 20
- Batch Size: 64

The training process involves iterating through the dataset, calculating the loss, and updating the model parameters using the Adam optimizer.

## Model Selection

To find the best model, hyperparameter tuning is performed by exploring different hidden layer sizes and dropout probabilities. The model with the highest test accuracy is selected and saved.

## Evaluation

The final trained model is evaluated on the test set, and metrics such as accuracy, F1 score, and recall are calculated. The classification report provides detailed information on precision, recall, and F1 score for each class.

## Results

The neural network demonstrates its capability to predict whether it will rain tomorrow based on the given weather features. The chosen hyperparameters lead to satisfactory model performance, and the evaluation metrics provide insights into the model's effectiveness.

## Next Steps

Further improvements and experimentation could involve exploring different architectures, conducting more extensive hyperparameter tuning, and investigating additional feature engineering techniques.

## Author

Ilaha Musayeva
10/13/2023
