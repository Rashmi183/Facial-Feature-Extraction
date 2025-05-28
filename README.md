# Facial Feature Extraction

## Overview
This project focuses on facial feature extraction using deep learning models, specifically **YOLO** and **Faster R-CNN**. It leverages the [Facial Feature Extraction Dataset](https://www.kaggle.com/datasets/osmankagankurnaz/facial-feature-extraction-dataset) from Kaggle to train and evaluate object detection models for identifying key facial features.

## Dataset
- **Source**: [Kaggle - Facial Feature Extraction Dataset](https://www.kaggle.com/datasets/osmankagankurnaz/facial-feature-extraction-dataset)
- **Description**: Contains labeled images with key facial landmarks, enabling model training for facial feature detection.

## Project Structure
```
Facial-Feature-Extraction/
│── Dataset/                        # Dataset used to train and test the model
│   │── train/                      # Training dataset
│   │── valid/                      # Validation dataset
│   │── test/                       # Testing dataset
│── Python Codes/                   # Scripts for training and evaluation
│   │── Facial_Feature_Extraction_EDA.ipynb  # Exploratory Data Analysis
│   │── YOLO_Model.ipynb            # YOLO training and evaluation
│   │── YOLO_Model_Evaluation.ipynb # YOLO model evaluation
│   │── RCNN_Final.ipynb            # Faster R-CNN training and evaluation
│── model/                         # Saved model weights
│   ├── best.pt                     # YOLO model weights
│── model_results/                  # Saved model results
│── data.yaml                       # Dataset configuration file
│── README.md                       # Project documentation
```

## Models Used
### YOLO (You Only Look Once)
- A real-time object detection model used for facial landmark detection.
- Trained on the given dataset with annotated facial features.

### Faster R-CNN (Region-based Convolutional Neural Network)
- A two-stage detector for high-accuracy object detection.
- Used to extract and identify key facial features.

## Installation & Setup
To run this project, install the required dependencies:
```bash
pip install torch torchvision numpy pandas opencv-python ultralytics
```

## Running the Models
### YOLO Model Training
```python
!python train.py --data data.yaml --weights yolovv8ns.pt --epochs 10 --imgsz=480 --device=0 --batch=16 --workers=8
```

### Faster R-CNN Model Training
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
```

## Evaluation
- Model evaluation notebooks analyze accuracy, precision, recall, and mAP scores.
- Visualizations show detected facial features on test images.

## Results
- **YOLO**: Achieved real-time performance with high precision.
- **Faster R-CNN**: Provided higher accuracy but required more computational resources.

## Deployed in Hugging Face
https://huggingface.co/spaces/manjuthiyagarajan2025/FacialFeatureExtraction
<img width="1492" alt="Screenshot 2025-03-16 at 10 16 10 AM" src="https://github.com/user-attachments/assets/f5604fac-2829-4638-aea1-00d5d74c6f66" />


## Future Improvements
- Fine-tuning hyperparameters for improved accuracy for Faster RCNN.
- Exploring transformer-based models for facial feature extraction.

## Authors
- Kapil Dixit, Manju Thiyagarajan, Rashmi Patel
