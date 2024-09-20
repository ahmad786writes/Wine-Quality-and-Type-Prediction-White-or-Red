# Wine Quality and Type Prediction (White or Red)

This repository implements a machine learning model using **TensorFlow** to predict the **quality** and **type** (white or red) of wine based on its physicochemical properties.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Introduction
The goal of this project is to predict two things about wine: 1. **Quality** - A score between 0 and 10 based on certain chemical properties. 2. **Type** - Whether the wine is **red** or **white**. The model is built using **TensorFlow** and trained on a dataset that contains several features describing the physicochemical properties of various wines.

## Dataset
The dataset used for this project is the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from the UCI Machine Learning Repository. The dataset consists of two different datasets: 1. **White wine dataset** (4898 samples) 2. **Red wine dataset** (1599 samples) Each wine sample contains the following features: - Fixed acidity - Volatile acidity - Citric acid - Residual sugar - Chlorides - Free sulfur dioxide - Total sulfur dioxide - Density - pH - Sulphates - Alcohol The **target variables** are: - **Quality** (regression task) - **Type** (classification task: red or white)

## Problem Statement
Given the physicochemical properties of wine, we aim to: 1. Predict the **quality** of the wine as a **regression** problem. 2. Classify the wine as either **red** or **white** as a **binary classification** problem.

## Architecture
The architecture for the model is based on a fully connected **Feedforward Neural Network (FNN)**. The network takes in the 11 features of the wine and outputs: 1. A **regression** output for predicting the quality score. 2. A **classification** output to predict whether the wine is red or white. 

### Model Structure
- Input Layer: 11 features (physicochemical properties)
- Dense Layers: Fully connected layers with ReLU activation
- Output Layer 1: Single neuron for **quality** prediction (regression)
- Output Layer 2: Single neuron with sigmoid activation for **type** prediction (binary classification)

### Loss Function
- For **quality** prediction: **Mean Squared Error (MSE)**.
- For **type** prediction: **Binary Crossentropy**.

### Optimizer
- **Adam** optimizer with learning rate adjustments.

## Training
The model is trained on both the **red wine** and **white wine** datasets, with labels for both **quality** and **type**. The training process uses backpropagation to minimize the combined loss (sum of MSE for quality and binary crossentropy for type).

## Evaluation
After training, the model is evaluated using: - **Mean Squared Error (MSE)** for the quality prediction. - **Accuracy** and **F1-Score** for the binary classification of wine type.

## Results
The model provides: - A **wine quality prediction** within a reasonable MSE. - A **wine type classification** with high accuracy (white or red).

### Example Predictions
- **Predicted Quality:** 6.7 (Actual: 7)
- **Predicted Type:** White (Correct)

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/wine-quality-prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook wine_quality_prediction.ipynb
    ```
4. Run the cells in the notebook to train and evaluate the model.

## References
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [TensorFlow Documentation](https://www.tensorflow.org/)
