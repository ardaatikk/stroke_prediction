# Importing clean data from dataset file
from train_test_datasets import train_X, train_y, val_X, val_y, train_data, test_data
from dataset import real_data

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Initializing and training the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_X, train_y)

# Making predictions on the training and validation data
train_predictions = model.predict_proba(train_X)[:, 1]
val_predictions = model.predict_proba(val_X)[:, 1]
"""test_predictions = model.predict(test_X)"""

# Calculating accuracy and classification report for the training data
train_accuracy = accuracy_score(train_y, train_predictions.round())
train_report = classification_report(train_y, train_predictions.round())

# Calculating accuracy and classification report for the validation data
val_accuracy = accuracy_score(val_y, val_predictions.round())
val_report = classification_report(val_y, val_predictions.round())

# Printing the training and validation results
print("Training Accuracy (without MinMaxScaler):", train_accuracy)
print("Training Classification Report (without MinMaxScaler):\n", train_report)
print("Validation Accuracy (without MinMaxScaler):", val_accuracy)
print("Validation Classification Report (without MinMaxScaler):\n", val_report)

# Scaling numerical features in real_data using MinMaxScaler
minmax = MinMaxScaler()
real_data[['age','avg_glucose_level','bmi']] = minmax.fit_transform(real_data[['age','avg_glucose_level','bmi']])
real_data = pd.get_dummies(real_data)

# Scaling numerical features in train_data using MinMaxScaler
minmax = MinMaxScaler()
train_data[['age','avg_glucose_level','bmi']] = minmax.fit_transform(train_data[['age','avg_glucose_level','bmi']])
train_data = pd.get_dummies(train_data)

# Checking for missing values in the train_data
missing_values_count_train = train_data.isnull().sum()
print("Train missing values count:", missing_values_count_train)

# Filtering outliers in 'avg_glucose_level' and 'bmi' columns in train_data
filtered_entries = np.array([True] * len(train_data))
for col in ['avg_glucose_level','bmi']:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - (IQR * 1.5)
    high_limit = Q3 + (IQR * 1.5)

    filtered_entries = ((train_data[col] >= low_limit) & (train_data[col] <= high_limit)) & filtered_entries    
train_data = train_data[filtered_entries]

# Printing the columns in train_data after preprocessing
print("Train data columns:", train_data.columns)

# Scaling numerical features in test_data using MinMaxScaler
minmax = MinMaxScaler()
test_data[['age','avg_glucose_level','bmi']] = minmax.fit_transform(test_data[['age','avg_glucose_level','bmi']])
test_data = pd.get_dummies(test_data)

# Printing the columns in test_data after preprocessing
print("Test data columns:", test_data.columns)

# Checking for missing values in the test_data
missing_values_count_test = test_data.isnull().sum()
print("Test missing values count:", missing_values_count_test)

print("Real data columns:", real_data.columns)

# Preparing the features and target for modeling using real_data
y = real_data.stroke
stroke_features =  ['age','avg_glucose_level',
       'bmi']

categorical_features = ['heart_disease', 'hypertension','gender_Female', 'gender_Male', 'gender_Other',
       'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
       'Residence_type_Urban', 'smoking_status_Unknown',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes' ]

"""test_stroke_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
       'bmi', 'gender_Female', 'gender_Male', 'gender_Other',
       'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
       'Residence_type_Urban', 'smoking_status_Unknown',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes']"""       

# Preparing the features and target for modeling using train_data and test_data
X = real_data[stroke_features]
train_X = pd.get_dummies(train_data[stroke_features + categorical_features])
test_X = pd.get_dummies(test_data[stroke_features + categorical_features])
train_y = train_data["stroke"]

# Aligning the train_X and test_X columns to ensure consistency
train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value=0)
train_X = train_X.astype(float)
test_X = test_X.astype(float)

# Splitting the train_X and train_y into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, random_state=0)

# Training the Random Forest Classifier on the training data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_X, train_y)

# Making predictions on the training and validation data
train_predictions = model.predict(train_X)
val_predictions = model.predict(val_X)
test_predictions = model.predict(test_X)

# Calculating accuracy and classification report for the training data
train_accuracy = accuracy_score(train_y, train_predictions)
train_report = classification_report(train_y, train_predictions)

# Calculating accuracy and classification report for the validation data
val_accuracy = accuracy_score(val_y, val_predictions)
val_report = classification_report(val_y, val_predictions)

# Printing the training and validation results
print("Training Accuracy (with MinMaxScaler):", train_accuracy)
print("Training Classification Report (with MinMaxScaler):\n", train_report)
print("Validation Accuracy (with MinMaxScaler):", val_accuracy)
print("Validation Classification Report (with MinMaxScaler):\n", val_report)

# Making predictions on the test data and saving the results to a CSV file
test_predictions = model.predict_proba(test_X)[:, 1]  # Probabilities of positive class (stroke)
test_data["stroke_prediction"] = test_predictions
selected_columns = ['id', 'stroke_prediction']
submission_data = test_data[selected_columns]
submission_data.to_csv("predictions.csv", index=False)
