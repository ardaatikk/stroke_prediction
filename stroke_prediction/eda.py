# Importing clean data from dataset file
from dataset import real_data

# Importing necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Extracting the unique values of gender column
gender = real_data["gender"].unique()
print("Unique gender values:", gender)

# Extracting the unique values of ever_married column
ever_married = real_data["ever_married"].unique()
print("Unique marriage values:", ever_married)

# Extracting the unique values of work_type column
work_type = real_data["work_type"].unique()
print("Unique work type values:", work_type)

# Extracting the unique values of Residence_type column
Residence_type = real_data["Residence_type"].unique()
print("Unique residence type values:", Residence_type)

# # Extracting the unique values of smoking_status column
smoking_status = real_data["smoking_status"].unique()
print("Unique smoking status values:", smoking_status)
# So we checked and there is no typos in these columns.

# Create a figure with subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 7))
fig.suptitle('Exploratory Data Analysis (Window 1)')

# Creating a boxplot to see the link between heart disease and age
sns.boxplot(x = real_data['heart_disease'], y = real_data['age'], palette= ["blue", "red"], ax=axs[0, 0])
axs[0, 0].set_title("Heart Disease vs Age") 

# Creating a barplot to see the link between heart disease and smoking status
sns.barplot(x = real_data['heart_disease'], y = real_data['smoking_status'], ax=axs[0, 1])
axs[0, 1].set_title("Heart Disease vs Smoking Status")

# Creating a barplot to see the link between smoking status and stroke
sns.barplot(x = real_data['stroke'], y = real_data['smoking_status'], ax=axs[0, 2])
axs[0, 2].set_title("Stroke vs Smoking Status")

# Creating a barplot to see the link between residence type and stroke
sns.barplot(x = real_data['Residence_type'], y = real_data['stroke'], ax=axs[1, 0])
axs[1, 0].set_title("Residence Type vs Stroke")

# Creating a lineplot to see the link between heart disease and bmi
sns.lineplot(x = real_data['heart_disease'], y = real_data['bmi'], ax=axs[1, 1])
axs[1, 1].set_title("Heart Disease vs BMI")
# The higher the bmi, the higher the likelihood of developing heart disease.

# Creating a boxplot to see the link between heart disease and average glucose level
sns.boxplot(x = real_data['heart_disease'], y = real_data['avg_glucose_level'], palette= ["blue", "red"], ax=axs[1, 2])
axs[1, 2].set_title("Heart Disease vs Avg Glucose Level")

# Creating a scatterplot with a jointdistribution plot to see the effect of the average glucose level and bmi on the stroke
sns.scatterplot(x = real_data['bmi'], y = real_data['avg_glucose_level'], hue = real_data['stroke'], ax=axs[2, 0])
axs[2, 0].set_title("BMI vs Avg Glucose Level")

# Creating a lineplot to see the link between heart disease and stroke
sns.lineplot(x = real_data['heart_disease'], y = real_data['stroke'], ax=axs[2, 1])
axs[2, 1].set_title("Heart Disease vs Stroke")

# Creating a boxplot to see the link between hypertension and age
sns.boxplot(x= real_data['hypertension'], y=real_data['age'], palette= ["blue", "red"], ax=axs[2, 2])
axs[2, 2].set_title("Hypertension vs Age")

plt.tight_layout()
plt.show()

# Create a figure with subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 7))
fig.suptitle('Exploratory Data Analysis (Window 2)')

# Creating a barplot to see the link between hypertension and heart disease
sns.barplot(x= real_data['hypertension'], y=real_data['heart_disease'], palette= ["blue", "red"], ax=axs[0, 0]) 
axs[0, 0].set_title("Hypertension vs Heart Disease")

# Creating a barplot to see the link between gender and stroke
sns.barplot(x = real_data['gender'], y = real_data['stroke'], palette= ["blue", "red"], ax=axs[0, 1])
axs[0, 1].set_title("Gender vs Stroke")

# Creating a lineplot to see the link between age and stroke
sns.lineplot(x = real_data['age'], y = real_data['stroke'], color = "red", ax=axs[0, 2])
axs[0, 2].set_title("Age vs Stroke")

# Creating a barplot to see the link between hypertension and stroke
sns.barplot(x = real_data['hypertension'], y = real_data['stroke'], palette= ["blue", "red"], ax=axs[1, 0])
axs[1, 0].set_title("Hypertension vs Stroke")

# Creating a barplot to see the link between heart disease and stroke
sns.barplot(x = real_data['heart_disease'], y = real_data['stroke'], palette= ["blue", "red"], ax=axs[1, 1])
axs[1, 1].set_title("Heart Disease vs Stroke")

# Creating a barplot to see the link between marriage and stroke
sns.barplot(x = real_data['ever_married'], y = real_data['stroke'], palette= ["blue", "red"], ax=axs[1, 2])
axs[1, 2].set_title("Ever Married vs Stroke")

# Creating a barplot to see the link work type disease and stroke
sns.barplot(x = real_data['stroke'], y = real_data['work_type'], ax=axs[2, 0])
axs[2, 0].set_title("Stroke vs Work Type")

# Creating a barplot to see the link between residence type and stroke
sns.barplot(x = real_data['Residence_type'], y = real_data['stroke'], palette= ["blue", "red"], ax=axs[2, 1])
axs[2, 1].set_title("Residence Type vs Stroke")

# Creating a violinplot to see the link between stroke and bmi
sns.violinplot(x = real_data['stroke'], y = real_data['bmi'], palette= ["blue", "red"], ax=axs[2, 2])
axs[2, 2].set_title("Stroke vs BMI")

plt.tight_layout()
plt.show()