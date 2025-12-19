# CIFAR-10 Image Classification using K-Nearest Neighbors

## Project Overview
This project implements an image classification system using the CIFAR-10 dataset and the K-Nearest Neighbors (KNN) machine learning algorithm. The objective is to classify images into predefined object categories based on visual similarity.

This is an academic machine learning project demonstrating supervised learning, distance-based classification, and model evaluation techniques.

## Problem Statement
How can we build a model that automatically identifies objects in images it has never seen before?

## Dataset
- Dataset Name: CIFAR-10
- Source: Kaggle CIFAR-10 Competition
- Total Images: 60,000
- Image Resolution: 32 x 32 pixels
- Number of Classes: 10

Classes include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

For experimentation, a balanced subset of 500 images per class was selected.

## Algorithm Used
### K-Nearest Neighbors (KNN)
KNN is a supervised learning algorithm that classifies a data point based on the majority class among its nearest neighbors.

Parameters tested:
- K values: 1, 3, 5, 7
- Distance metrics:
  - Euclidean
  - Manhattan
  - Cosine

## Methodology
1. Loaded CIFAR-10 data from CSV files
2. Performed balanced sampling of classes
3. Flattened image data into feature vectors
4. Split dataset into 80 percent training and 20 percent testing data
5. Trained KNN models using different K values and distance metrics
6. Evaluated performance using accuracy and confusion matrix

## Results
- Best accuracy achieved: approximately 30.60 percent
- Best configuration:
  - Distance metric: Manhattan
  - K value: 5

The model performed reasonably given the high dimensionality of image data, though it struggled with visually similar classes.

## Limitations
- Poor scalability with large datasets
- High computation cost for distance calculations
- Limited performance on high-dimensional image features

## Future Scope
- Apply dimensionality reduction techniques such as PCA
- Implement CNN-based deep learning models
- Perform hyperparameter tuning with cross-validation
- Increase dataset size and apply data augmentation

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Project Structure
- Cifar_10.ipynb
- trainLabels.csv
- test.csv
- IBM Project.pptx
- README.md
