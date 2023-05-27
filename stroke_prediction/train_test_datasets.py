# Importing clean data from dataset file
from dataset import real_data

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Reading the train and test files into dataframes
train_data = pd.read_csv("/Users/ardaatik/Downloads/project/data/train.csv")
test_data = pd.read_csv("/Users/ardaatik/Downloads/project/data/test.csv")

# Categorizing age values into different age groups
train_data['age_bin'] = pd.cut(train_data['age'], bins=[0, 18, 40, 60], labels=['young', 'middle-aged', 'elderly'])
test_data['age_bin'] = pd.cut(test_data['age'], bins=[0, 18, 40, 60], labels=['young', 'middle-aged', 'elderly'])

# Categorizing bmi values into different bmi groups
train_data['bmi_category'] = pd.cut(train_data['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal weight', 'overweight', 'obese'])
test_data['bmi_category'] = pd.cut(test_data['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal weight', 'overweight', 'obese'])

# Creating an interaction feature between 'age' and 'hypertension' and adding it to the train and test data
train_data['age_hypertension_interaction'] = train_data['age'] * train_data['hypertension']
test_data['age_hypertension_interaction'] = test_data['age'] * test_data['hypertension']

# Extracting the target variable 'stroke' from the real_data
y = real_data["stroke"]

# Selecting the features for modeling
stroke_features = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
categorical_features = ["smoking_status", "ever_married"]

# Extracting the features 'stroke_features' from the real_data
X = real_data[stroke_features]

# Splitting the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Creating a new feature set 'stroke_features_with_interaction' with the original features and the interaction feature
stroke_features_with_interaction = stroke_features + ['age_hypertension_interaction']

# One-hot encoding categorical features and combining them with the interaction and original features for training and test data
train_X = pd.get_dummies(train_data[stroke_features_with_interaction + categorical_features])
test_X = pd.get_dummies(test_data[stroke_features_with_interaction + categorical_features])

train_y = train_data["stroke"]
# Aligning the columns of train_X and test_X to ensure they have the same features for modeling
train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value=0)

# Converting train_X and test_X data to float type for modeling
train_X = train_X.astype(float)
test_X = test_X.astype(float)

# Splitting the training data further into train and validation sets
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, random_state=0)

# Converting train_X and test_X data to float type for modeling (after the second split)
train_X = train_X.astype(float)
test_X = test_X.astype(float)