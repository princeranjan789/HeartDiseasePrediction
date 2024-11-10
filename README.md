# Heart Disease Prediction Project

This project uses machine learning techniques to predict the likelihood of heart disease based on various health indicators. The goal is to build a reliable predictive model to assist healthcare providers in identifying individuals at high risk of heart disease, potentially aiding in early diagnosis and preventive care.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Implementation](#3-model-implementation)
  - [4. Results and Discussion](#4-results-and-discussion)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

Heart disease is one of the leading causes of death worldwide, and early detection is crucial for effective treatment and improved outcomes. This project involves building a classification model that predicts the presence of heart disease based on a set of health indicators, such as age, cholesterol levels, and chest pain type.

## Dataset

The dataset used for this project contains medical information from patients, with various health metrics and indicators as features. Key columns include:

- **Age**: Age of the patient
- **Sex**: Gender of the patient (`M` for male, `F` for female)
- **ChestPainType**: Type of chest pain (e.g., `Typical Angina`, `Atypical Angina`)
- **RestingBP**: Resting blood pressure (in mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dl)
- **FastingBS**: Fasting blood sugar level
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (`Y` for yes, `N` for no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: The slope of the peak exercise ST segment
- **HeartDisease**: Target variable (`1` indicates presence of heart disease, `0` indicates absence)

Additional features were engineered based on these existing attributes to capture complex interactions.

## Approach

### 1. Exploratory Data Analysis (EDA)
   - **Goal**: To understand feature distributions, relationships, and identify patterns or trends.
   - **Techniques**: Visualizations using histograms, scatter plots, heatmaps for correlations, and count plots.
   - **Key Insights**: Identified key predictors of heart disease such as age, cholesterol, chest pain type, and maximum heart rate.

### 2. Feature Engineering
   - **New Features**: Created interaction terms like `age_cholesterol_interaction` to capture the combined effect of multiple health indicators.
   - **Encoding**: Applied one-hot encoding for multi-class categorical features (e.g., `ChestPainType`) and label encoding for binary features (e.g., `Sex`).
   - **Feature Scaling**: Standardized numerical features such as age, resting BP, and cholesterol to ensure uniformity across different scales.

### 3. Model Implementation
   Several machine learning models were tested, including:
   - **Random Forest Classifier**: The best-performing model, capable of capturing complex relationships between features.
   - **Logistic Regression**: A baseline model with strong interpretability.
   - **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** for comparison.

   Each model was evaluated using metrics like accuracy, F1 score, and ROC AUC.

### 4. Results and Discussion
   - **Best Model**: The Random Forest model achieved the highest accuracy and robustness, with an ROC AUC score of 0.90.
   - **Important Predictors**: Key predictors include age, cholesterol, maximum heart rate, and chest pain type.
   - **Limitations**: While the model performs well, the dataset could be expanded to improve generalizability across more diverse populations.

## Results

| Model                 | Accuracy | F1 Score | ROC AUC |
|-----------------------|----------|----------|---------|
| Random Forest         | 0.88     | 0.85     | 0.90    |
| Logistic Regression   | 0.85     | 0.83     | 0.87    |
| Support Vector Machine | 0.86     | 0.84     | N/A     |
| K-Nearest Neighbors   | 0.82     | 0.81     | 0.83    |

The project showed that certain health indicators are strong predictors of heart disease, providing insights that can be valuable for healthcare providers.

## Technologies Used

- **Python**: Core programming language.
- **Pandas, NumPy**: Data manipulation and analysis.
- **Matplotlib, Seaborn**: Data visualization.
- **Scikit-learn**: Model training, validation, and evaluation.

## Getting Started

### Prerequisites

- Python 3.7 or above
- Required Libraries: Install dependencies using `pip install -r requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/princeranjan789/HeartDiseasePrediction.git
   cd HeartDiseasePrediction
