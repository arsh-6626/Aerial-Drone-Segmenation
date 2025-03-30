# Aerial Drone Segmentation

## Overview

This project focuses on performing semantic segmentation on aerial drone images using deep learning techniques. Semantic segmentation involves classifying each pixel in an image into a specific category, which is essential for applications such as land mapping, urban planning, and environmental monitoring.

## Repository Structure

- `dataloader.py`: Contains functions for loading and preprocessing the dataset.
- `train.py`: Script to train the segmentation model.
- `unet.py`: Implementation of the U-Net architecture for semantic segmentation.

## Dataset

The project utilizes the [Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset), which provides high-resolution aerial images captured from drones. This dataset is widely used in the field of aerial image segmentation.

## Model Architecture

The U-Net architecture is employed for segmentation tasks. U-Net is a convolutional neural network designed for biomedical image segmentation but has proven effective in various segmentation tasks, including aerial imagery.

## Usage

1. **Dataset Preparation**: Download the Semantic Drone Dataset and organize it in the appropriate directory structure.
2. **Training**: Execute `train.py` to train the U-Net model on the dataset.
3. **Evaluation**: Assess the model's performance using standard segmentation metrics.

We appreciate the contributions of the authors and the open-source community for providing valuable resources that have significantly aided this project.
