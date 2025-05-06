🩺 Liver Cirrhosis Stage Prediction 🔬
This project leverages machine learning to predict the stage of liver cirrhosis based on a variety of medical features. Liver cirrhosis is a severe liver condition where the liver becomes scarred over time, often due to chronic diseases. Early detection of cirrhosis stages is vital for timely treatment and management. Using a Random Forest Classifier, this project predicts the cirrhosis stage from a set of clinical attributes, empowering healthcare professionals to make informed decisions.

📑 Table of Contents
Project Overview

Dependencies

Data Description

Model Building

Training the Model

Prediction Interface

Evaluation

Usage

License

🧑‍⚕️ Project Overview
This project builds a machine learning model that predicts the stage of liver cirrhosis using patient data. The model uses a Random Forest Classifier, which is highly effective for both numerical and categorical data, to predict the cirrhosis stage based on various medical features.

Steps Involved:
Data Loading & Preprocessing:

We load the data from a CSV file and encode categorical variables into numerical values, making them suitable for machine learning.

Model Training:

The data is split into training and testing sets, and the Random Forest Classifier is trained to identify relationships between features and the target variable (Cirrhosis Stage).

Prediction:

A simple prediction interface allows users to input new patient data and get immediate predictions for the cirrhosis stage.

Evaluation:

The model is evaluated using accuracy, helping us understand how well the model performs on unseen data.

📦 Dependencies
To get started with the project, you'll need the following Python packages:

pandas: Data manipulation and analysis.

numpy: Numerical operations.

scikit-learn: For machine learning algorithms and utilities.

ipywidgets: For interactive widgets in Jupyter notebooks.

matplotlib (optional): For plotting and data visualization.

Installation Command:
bash
Copy
Edit
pip install pandas numpy scikit-learn ipywidgets matplotlib
📊 Data Description
The dataset used in this project contains various columns representing medical attributes and a target column indicating the cirrhosis stage.

Features:
N_Days: Days since diagnosis.

Status: Current health status (C for Cirrhosis, T for Transplant).

Drug: Medication prescribed.

Age: Age of the patient (in days).

Sex: Gender (M for Male, F for Female).

Ascites: Presence of fluid retention in the abdomen.

Hepatomegaly: Presence of an enlarged liver.

Spiders: Presence of spider angiomas.

Edema: Presence of swelling.

Bilirubin: Bilirubin levels.

Cholesterol: Cholesterol levels.

Albumin: Albumin protein levels.

Copper: Copper levels.

Alk_Phos: Alkaline phosphatase levels.

SGOT: Serum glutamic-oxaloacetic transaminase levels.

Triglycerides: Triglyceride levels.

Platelets: Platelet count.

Prothrombin: Blood clotting ability.

Target Variable:
Stage: The cirrhosis stage (e.g., Stage 1, Stage 2, etc.).

🧑‍💻 Model Building
Feature Engineering:
We carefully selected features that are closely related to liver function and cirrhosis progression. These include clinical symptoms like ascites, hepatomegaly, and lab results like bilirubin levels.

Label Encoding:
Categorical variables such as Status, Sex, and Ascites are encoded to numerical values using LabelEncoder.

Random Forest Classifier:
The model uses Random Forest, an ensemble method that builds multiple decision trees and combines their results to make a robust and accurate prediction.

🛠️ Training the Model
Data Splitting:
The data is split into training (80%) and testing (20%) sets to evaluate how well the model generalizes to unseen data.

Model Training:
The Random Forest Classifier is trained using the fit() method, where the model learns patterns from the training data and adjusts its parameters accordingly.

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset
df = pd.read_csv('liver_cirrhosis.csv')

# Split into features (X) and target (y)
X = df.drop('Stage', axis=1)
y = df['Stage']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical features
encoders = {}
for col in X_train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    encoders[col] = le

# Train the Random Forest model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Evaluate the model's performance
ypred = rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, ypred))
🎮 Prediction Interface
Using ipywidgets, the model provides an interactive interface where users can input patient data and instantly get predictions on the liver cirrhosis stage.

python
Copy
Edit
import ipywidgets as widgets
from ipywidgets import interact
import pandas as pd

# Sample input for user interface
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

# Create the interactive widget
interact(make_prediction, **input_dict)
📊 Evaluation
The model's accuracy is evaluated using the test dataset. We calculate the accuracy score, which reflects the percentage of correct predictions on unseen data. This helps us gauge the model's generalization ability.

🚀 Usage
How to Use:
Install the necessary dependencies:

bash
Copy
Edit
pip install pandas numpy scikit-learn ipywidgets matplotlib
Download the liver_cirrhosis.csv dataset and place it in the same directory as your code.

Run the training script to train the model on the dataset.

Use the interactive interface to input data and receive real-time predictions for the cirrhosis stage.


