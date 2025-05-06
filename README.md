Liver Cirrhosis Stage Prediction
This project aims to predict the stage of liver cirrhosis based on various medical features using machine learning. Liver cirrhosis is a condition where the liver becomes severely scarred over time, often due to chronic liver disease. The early detection of cirrhosis stages is crucial for treatment and patient management. This machine learning model uses a Random Forest Classifier to predict the stage of liver cirrhosis based on various patient attributes, including biochemical markers and clinical symptoms.

Table of Contents
Project Overview

Dependencies

Data Description

Model Building

Training the Model

Prediction Interface

Evaluation

Usage

License

Project Overview
The liver cirrhosis stage prediction project builds a machine learning model that predicts the stage of liver cirrhosis based on medical data. The dataset consists of various features related to liver function and the presence of clinical conditions that contribute to the progression of cirrhosis.

The goal of this project is to classify liver cirrhosis into predefined stages (e.g., Stage 1, Stage 2, etc.) based on a set of medical attributes, enabling timely and effective medical intervention.

Steps Involved:
Data Loading and Preprocessing:

The data is loaded from a CSV file and categorical variables such as Status, Sex, Ascites, etc., are encoded into numerical values for machine learning compatibility.

Categorical columns are encoded using Label Encoding to convert non-numeric data into a form that machine learning algorithms can understand.

Model Training:

The dataset is split into training and testing sets, and a Random Forest Classifier is trained to learn patterns between medical features and the target variable (Stage).

The model is then trained using the RandomForestClassifier algorithm, which builds multiple decision trees and combines their results for more accurate predictions.

Prediction:

A user-friendly interface is created, allowing the user to input patient data and receive predictions about the liver cirrhosis stage.

Using ipywidgets, the interface allows for easy input and dynamic predictions based on user data.

Evaluation:

The model's accuracy is evaluated using the test data, and performance metrics like accuracy are calculated.

Accuracy score helps in understanding the reliability of the model when used on unseen data.

Dependencies
Ensure that the following Python packages are installed to run this project:

pandas: For data manipulation and analysis.

numpy: For numerical operations.

scikit-learn: For machine learning models and utilities like model evaluation.

ipywidgets: For creating interactive widgets in Jupyter notebooks.

matplotlib (optional): For data visualization and plotting.

To install the dependencies, you can run the following command:

bash
Copy
Edit
pip install pandas numpy scikit-learn ipywidgets matplotlib
Data Description
The dataset (liver_cirrhosis.csv) used in this project contains several columns that represent medical features, and the target column represents the liver cirrhosis stage.

Features:
N_Days: Number of days since the patient’s diagnosis of cirrhosis.

Status: Current health status of the patient (C for cirrhosis, T for transplant).

Drug: The drug prescribed to the patient (Placebo, DrugA, DrugB).

Age: The age of the patient (in days).

Sex: Gender of the patient (M for Male, F for Female).

Ascites: Whether the patient has ascites (fluid retention in the abdomen) (Y for Yes, N for No).

Hepatomegaly: Whether the patient has hepatomegaly (enlarged liver) (Y for Yes, N for No).

Spiders: Presence of spider angiomas (a symptom of liver disease) (Y for Yes, N for No).

Edema: Presence of edema (swelling) (Y for Yes, N for No).

Bilirubin: Blood bilirubin levels, which are often elevated in liver disease.

Cholesterol: Cholesterol levels in the blood.

Albumin: The albumin protein level in the blood, which is typically low in liver disease.

Copper: Copper levels in the blood.

Alk_Phos: Alkaline phosphatase levels, which can indicate liver damage.

SGOT: Serum glutamic-oxaloacetic transaminase, an enzyme elevated in liver disease.

Triglycerides: Blood triglyceride levels.

Platelets: Platelet count in the blood.

Prothrombin: A measure of blood clotting ability, often impaired in liver disease.

Target Variable:
Stage: The stage of liver cirrhosis, which is the target variable for prediction (Stage 1, Stage 2, etc.).

Model Building
Feature Selection:
The features used in this project were selected based on their relevance to liver cirrhosis progression. Medical attributes related to liver function, such as albumin, bilirubin, and cholesterol, are included, as they are known to affect liver health.

Label Encoding:
Categorical features like Status, Sex, Ascites, etc., are encoded into numerical values using LabelEncoder to convert categorical data into a form suitable for machine learning models.

Random Forest Classifier:
The model chosen for this project is the Random Forest Classifier. This ensemble model combines the results from multiple decision trees to provide more robust predictions. Random Forests work well for both numerical and categorical data, making it a suitable choice for this task.

Model Evaluation:
The model's performance is evaluated using accuracy, which is calculated by comparing the predicted stages with the actual values in the test set.

Training the Model
Data Split:
The dataset is split into training and testing sets using train_test_split. The training data is used to train the model, while the test data is used to evaluate the model's performance.

Model Training:
The Random Forest Classifier is trained on the features (X) and target (y) using the fit() method. During training, the model learns the relationship between input features and the output target (Stage).

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
df = pd.read_csv('liver_cirrhosis.csv')

# Split data into features (X) and target (y)
X = df.drop('Stage', axis=1)
y = df['Stage']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical columns
encoders = {}
for col in X_train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    encoders[col] = le

# Train Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Evaluate the model
ypred = rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, ypred))
Prediction Interface
To make the model easily usable, a simple prediction interface is created using ipywidgets in Jupyter notebooks. This allows users to input patient data and receive predictions for the liver cirrhosis stage.

python
Copy
Edit
import ipywidgets as widgets
from ipywidgets import interact
import pandas as pd

# Input fields for user
input_dict = {
    'N_Days': 1230,
    'Status': 'C',
    'Drug': 'Placebo',
    'Age': 19724,
    'Sex': 'M',
    'Ascites': 'Y',
    'Hepatomegaly': 'N',
    'Spiders': 'Y',
    'Edema': 'N',
    'Bilirubin': 0.5,
    'Cholesterol': 219.0,
    'Albumin': 3.93,
    'Copper': 22.0,
    'Alk_Phos': 663.0,
    'SGOT': 45.0,
    'Tryglicerides': 75.0,
    'Platelets': 220.0,
    'Prothrombin': 10.8
}

# Function to make a prediction
def make_prediction(**kwargs):
    input_df = pd.DataFrame([kwargs])
    for col, le in encoders.items():
        input_df[col] = input_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]
    prediction = rfc.predict(input_df)
    print("Predicted Stage:", prediction[0])

# Create widgets
interact(make_prediction, **input_dict)
Evaluation
After training the model, its performance is evaluated by comparing the predicted values to the actual test data. The accuracy score is calculated, indicating the percentage of correctly predicted stages.

Usage
Steps to use the project:
Install the necessary dependencies using the command:

bash
Copy
Edit
pip install pandas numpy scikit-learn ipywidgets matplotlib
Download the liver_cirrhosis.csv dataset and place it in the same directory as your script.

Run the model training script to train the model on the dataset.

Use the interactive prediction interface to input a patient's data and get the predicted cirrhosis stage.


