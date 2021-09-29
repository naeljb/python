# importing the necessary libraries 
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# loading my dataset?
d = "C:/Users/Nael/Desktop/zambezia.csv"  # providing the dataset path
df = pd.read_csv(d)    # reading the csv file and saving as df 
df

# Checking the data type in each column
df.dtypes

# Checking the name of each column
df.columns.values

#  adding a colunn for 70% of population 
df['pop_target']=round(df['population ']*0.70)

# adding a colunn for  number of  households 
df['household'] = round(df['pop_target']/5)

# adding a colunn for number of males
df['male'] = round(df['pop_target']*0.48)

# adding a colunn for number of females
df['female'] = round(df['pop_target']*0.52 )

# adding a colunn for number of boy < 5
df['boy'] = round(df['pop_target']*0.0853 )

# adding a colunn for number of girl < 5
df['girl'] = round(df['pop_target']*0.0827 )

# adding a colunn for number of children < 5
df['u5'] = round(df['girl'] +df['boy'])

# adding a colunn for number of boy < 2
df['u2'] = round(df['u5']*0.40 )

df

tot_population= df['population '].sum()
tot_population

