# importing the necessary libraries 
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# KNOW YOUR DATASET

# How to load my dataset?
d = "https://raw.githubusercontent.com/naeljb/python/main/rawdata.csv"  # providing the dataset path
df = pd.read_csv(d)    # reading the csv file and saving as df 

# How to view the first five rows of my dataset?
df.head()

# How to view the last five rows of my dataset?
df.tail()

# What is the size of my dataset (number of rows and columns)?
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

# How to create new dataset which is a subset of columns of another dataset ?
subset =df[['district','sex','age','hh_size']]

# how to input datset directly?   Here I'm inputing two datasets
d1 = {'c':[1,2,3,4,5,8,15,45,7,3],'b':[5,6,7,8,5,6,7,8,9,10],'income':[9,10,11,12,5,6,7,9,5,45]}
pre_test =pd.DataFrame(d1)
pre_test

d2 = {'income':[1,2,3,4,7,9,6,45,5,12],'b':[5,6,7,8,89,5,74,90,22,11],'c':[9,10,11,12,1,6,7,8,9,11]}
post_test =pd.DataFrame(d2)
post_test

# how to re-order the columns?  here I'm re-ordering df1 dataset so that later I can append it to df2 dataset
pre_test=pre_test[['income','b','c']]
pre_test

# how to append two datasets?
df3 = pd.concat([pre_test,post_test])
df3

# how to drop columns?
df=df.drop(['program','test_registration##recipient_link_key'], axis=1)

# how to remove rows that met certain conditions? can be used for both sorting and cleaning 
#  Here I'm  removing all people aged between 18-24 years old  in miami 
df =df[(df['age']>=18) & (df['age']<=24) & (df['city']!='miami')]

# how to rename columns?  
df.rename(columns={'recipient_name':'name','account_number':'account','vulnerability_level':'vulnerability','hh_size':'size','household_status':'status'},inplace=True)

# TRANSFORM YOUR DATA

# How to replace missing values in a column by the mean ?
mean_age = df['age'].mean(axis=0)  # Calculation the mean for age variable
df['age'].replace(np.nan,mean_age,inplace =True)

# how to replace missing values in a column by the most frequent value in that column? 
df['district'].value_counts() # Counting number of each value label in that column 
df['district'].value_counts().idxmax() # identifying the value label with the maximun count(for my dataset, it will return 'est') 
df['district'].replace(np.nan,'est',inplace =True)

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

# How to save my dataset with the Extracted,transformed into a directory? 
df.to_csv('C:/Users/Nael/Desktop/mydataset.csv') # here saving my Desktop as a CSV file

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

# how to do a cross tabublation (count) betwenn two columns ?  
pd.crosstab(df['district'], df['sex'])

# How to do a cross tabulation (count ) bar chart betwenn two columns? 
pd.crosstab(df['district'], df['sex']).plot(kind = 'bar')
plt.title('cross tabulation figure')
plt.show()

# how to do a cross tabublation (percentage) betwenn two columns ?
pd.crosstab(df['district'], df['sex'], normalize ='all') #  (% over all value,normalize overall values )

pd.crosstab(df['district'], df['sex'], normalize ='index') #  (% or normalize over each row )

pd.crosstab(df['district'], df['sex'], normalize ='columns')  #  (% or normalize over each column )

# How to create a subset (from a value of column) and getting the summary statistics of it?
male_df = df[df['sex']=='Male']  # creating my subset of male
male_df.describe()

# (here statiscts for "female" group of "sex" column)
female_df = df[df['sex'] == 'Female']
female_df.describe()

# Are the means of two groups within a same variable/column satisticaly different?

from scipy.stats import ttest_ind   # importing  independant t-test function   

TwoTail = ttest_ind(male_df['age'],female_df['age'],equal_var = True)
TwoTail  # result nterpretation: if p-value less than 0.05, means are statistically different

# Are the means taken on the same group but at two different times statistically different?
income_before = pre_test['income'] # extracting income data before  from pre_test dataset
income_after = post_test['income'] # extracting income data after from post_test dataset

from scipy.stats import ttest_rel  # importing paired t-test function

ttest_pair = ttest_rel(income_before,income_after)
ttest_pair  # result nterpretation: if p-value less than 0.05, means are statistically different

deg_free =(len(income_before) + len(income_after)) - 1  # degree of freedom for the paired t-test
deg_free

# Are there correlation among the variables (option 1: table )?
df.corr()  #

# Are there correlation among the variables (option 2: heatmap  )?
heatmap= sns.heatmap(df.corr(),cmap = 'Blues', annot = True)
heatmap.set_title('Correlation heatmap', pad =12)
plt.show()
