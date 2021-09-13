# importing the necessary libraries 
import pandas as pd
import numpy as np
import math
import scipy.stats as stats

# KNOW YOUR DATASET

# How to load a dataset?
d = "https://raw.githubusercontent.com/naeljb/python/main/rawdata.csv"  # providing the dataset path
df = pd.read_csv(d)    # reading the csv file and saving as df 

# How to view the first five rows?
df.head()

# How to view the last five rows?
df.tail()

# What is the dataset size (number of rows and columns)?
df.shape

# How to get the column headers?
df.columns.values

# What is the data type of each  column?
df.dtypes

# What are the different value names in a column (ex: in district column)?
df['district'].unique()

# Are there missing values in a column and how many?
df.isnull().sum()
