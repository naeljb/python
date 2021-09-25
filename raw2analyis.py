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

# how to drop columns?
df=df.drop(['program','test_registration##recipient_link_key'], axis=1)

# how to remove rows that met certain conditions? can be used for both sorting and cleaning 
#  Here I'm  removing all people aged between 18-24 years old  in miami 
df =df[(df['age']>=18) & (df['age']<=24) & (df['city']!='miami')]

# how to rename columns?  
df.rename(columns={'recipient_name':'name','account_number':'account','vulnerability_level':'vulnerability','hh_size':'size','household_status':'status'},inplace=True)

# TRANSFORM YOUR DATA

# how to add a new column by applying an operation on another column? 
df['indirect_bene']=df['size']*5

# how to add a new column with a single value 
df['beneficiaries'] = '1'

# how to create a new column with a single condition applied on another column (option 1)?
df['adulte']= df['age'].apply(lambda x: 'yes ' if x >18 else 'no')

# how to create a new column with a single condition applied on another column (option 2)?
def age_range(a):
    if a['age'] > 18:
        return 1
    else:
        return 0
df ['adulte1'] =df.apply(age_range,axis=1)

# how to create a new column with multiple conditions applied on another column?
def age_range1(b):
    if b['age'] > 11 and b['age']<30:
        return 1
    else:
        return 0

df ['youth'] =df.apply(age_range1,axis=1)

# how to create a new column by changing the value labels of another column?
df['sex_num']=df.sex.map({'Female':0,'Male':1})

# ANALYZE YOUR DATA

# how to get summary statistics?
df.describe()

# how to get the upper and lower 95% intervall confidence values (option 1)?
n = df['age']. count() # counting number of values in the column
nsquart =math.sqrt(n)   # squar of n
mean = df['age'].mean()
std = df['age']. std()
z = stats.norm.ppf(q=0.975)
upper = mean +(z*(std/nsquart))
lower = mean -(z*(std/nsquart))
print(upper,lower)

# how to get the upper and lower 95% intervall confidence values (option 2)?
u_age = df['age'].mean() + (stats.norm.ppf(q=0.975)*(df['age']. std()/math.sqrt(df['age']. count())))
l_age = df['age'].mean() - (stats.norm.ppf(q=0.975)*(df['age']. std()/math.sqrt(df['age']. count())))
print (u_age,l_age)

# how to get the percentage of each value label in a column?
s = df['sex']
counts= s.value_counts()
percents=s.value_counts(normalize=True)
percents100= s.value_counts(normalize=True).mul(100).round(1).astype(str)+"%"
d2=({'#':counts,'percent':percents,'%':percents100})
sex1=pd.DataFrame(d2)
sex1

