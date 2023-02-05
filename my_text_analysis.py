# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:42:24 2023

@author: Nael Jean-Baptiste
"""
# IMPORTING THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')

# READING THE DATASET 
d = "https://raw.githubusercontent.com/naeljb/python/main/my_monkey_survey.csv"
df = pd.read_csv(d)  

# IMPLEMENTING BAG OF WORDS

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Get the list of stop words for English
stop_words = set(stopwords.words('english'))

# Define a function to filter out stop words
def remove_stopwords(response):
    words = word_tokenize(response)
    filtered_words = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered_words)

# Apply the function to the "text" column
df["text_filtered"] = df["response"].apply(remove_stopwords)

from nltk.probability import FreqDist

# Tokenize the text in the "text_filtered" column
df["text_fitered_tokens"] = df["text_filtered"].apply(nltk.word_tokenize)

# Get frequency distribution of the words in the "text_filtered_tokens" column
fdist = FreqDist(df["text_fitered_tokens"].sum())

# Plot the frequency distribution
fdist.plot(10, cumulative=False)
plt.show()

# IMPLEMENTING WORD CLOUD

from wordcloud import WordCloud
from PIL import Image

# Load image for the background (use an image stored in your local machine)
mask = np.array(Image.open("C:/Users/Nael/Desktop/logo1.jpg"))

# Combine all the texts in 'text_column'
text = " ".join(review for review in df['response'])

# Create the word cloud object
wordcloud = WordCloud(background_color="white", max_words=200, mask=mask)
                     
# Generate a word cloud
wordcloud = wordcloud.generate(text)

# Plot the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.show()

# IMPLEMENTING SENTIMENT ANALYSIS 

# importing the sentiment analyzer module
from nltk.sentiment import SentimentIntensityAnalyzer

# Add a new column named "sentiment" to the dataframe
df["sentiment"] = None

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Iterate through the rows of the dataframe and use the polarity_scores()
# method to get the sentiment of the text
for index, row in df.iterrows():
    sentiment = sia.polarity_scores(row["response"])
    df.at[index, "sentiment"] = sentiment["compound"]
    
# Computing the average sentiment score and statictics descriptive
print( df["sentiment"].describe())
sentiment_mean = df["sentiment"].mean()

# let's have sentiment of guest 4 (it will return a dictionary)
s= sia.polarity_scores("sorry for being late and could not enjoy all the food")
print(s)

# Plotting sentiment score
plt.hist(df["sentiment"], bins=20, edgecolor='black')
plt.xlabel('Sentiment score')
plt.ylabel('Frequency')


