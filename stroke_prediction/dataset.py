# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

### Getting the real data
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate back one level to the parent directory
parent_dir = os.path.dirname(current_dir)

# Define the relative path to the data file
data_folder = os.path.join(parent_dir, "data")
data_file_path = os.path.join(data_folder, "healthcare-dataset-stroke-data.csv")

# Read the CSV file
real_data = pd.read_csv(data_file_path)
real_data.head()# 5110 row x 12 column

# Checking the missing values
missing_values_count = real_data.isnull().sum()
if __name__ == "__main__":
      print("Missing values:", missing_values_count) # only bmi 201 values are missing

# Calculating the missing percent
total_cells = np.product(real_data.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing/total_cells) * 100
if __name__ == "__main__":
      print("Percent missing:", percent_missing)

# Filling the missing values
real_data.fillna(method='bfill', axis=0).fillna(0)

# get the index of all positive pledges (Box-Cox only takes positive values)
index_of_positive_pledges = real_data.avg_glucose_level > 0

# get only positive pledges (using their indexes)
positive_pledges = real_data.avg_glucose_level.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                               name='avg_glucose_level', index=positive_pledges.index)

if __name__ == "__main__":
      print('Original data\nPreview:\n', positive_pledges.head())
      print('Minimum value:', float(positive_pledges.min()),
            '\nMaximum value:', float(positive_pledges.max()))
      print('_'*30)

      print('\nNormalized data\nPreview:\n', normalized_pledges.head())
      print('Minimum value:', float(normalized_pledges.min()),
            '\nMaximum value:', float(normalized_pledges.max()))

      # Plotting histogram
      ax = sns.histplot(normalized_pledges, kde=True)
      ax.set_title("Normalized data")
      plt.show()