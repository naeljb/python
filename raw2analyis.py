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

# SHAPE YOUR DATASET

# how to input datset directly?   Here I'm inputing two datasets
d1 = {'c':[1,2,3,4],'b':[5,6,7,8],'a':[9,10,11,12]}
df1 =pd.DataFrame(d1)
df1

d2 = {'a':[1,2,3,4],'b':[5,6,7,8],'c':[9,10,11,12]}
df2 =pd.DataFrame(d2)
df2

# how to re-order the columns?  here I'm re-ordering df1 dataset so that later I can append it to df2 dataset
df1=df1[['a','b','c']]
df1

# how to append two datasets?
df3 = pd.concat([df1,df2])
df3

# how to drop columns?
df=df.drop(['program','test_registration##recipient_link_key'], axis=1)

# how to remove rows that met certain conditions? can be used for both sorting and cleaning 
#  Here I'm  removing all people aged between 18-24 years old  in miami 
df =df[(df['age']>=18) & (df['age']<=24) & (df['city']!='miami')]

# how to rename columns?  
df.rename(columns={'recipient_name':'name','account_number':'account','vulnerability_level':'vulnerability','hh_size':'size','household_status':'status'},inplace=True)
df.head(2)
