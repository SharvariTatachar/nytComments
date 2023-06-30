import pandas as pd 
import numpy as np 
import re 
import string 
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
import nltk
stemmer = nltk.SnowballStemmer("english")
stopword = stopwords.words('english')


#READING IN ARTICLE DETAILS ------------

#only using articleID, headlines, newDesk, keywords:
articles = pd.read_csv("/Users/sharrutatachar/Desktop/ArticlesApril2018.csv", 
usecols = list(range(0,1)) + list(range(4,8)))
del articles['multimedia']

#truncate dataset to relevant articles for smaller set: 
lst_topics = ["Politics", "Culture", "National"]
art_df = articles.query('newDesk in @lst_topics') 

#READING IN COMMENT DETAILS ------------

#using articleID, commentBody, and commentID:
comments = pd.read_csv("/Users/sharrutatachar/Desktop/CommentsApril2018.csv",
usecols = list(range(1,2)) + list(range(3,5)))

#get the comments that are relevant to the chosen articles: 
articleIDs = art_df['articleID'].values.tolist() 
comments_general = comments.query('articleID in @articleIDs') #around 24k comments


#CLEAN COMMENT BODY ---------------

def clean(comment): 
    comment = str(comment).lower()
    comment = re.sub('<br/><br/>', '', comment)
    comment = re.sub('<br/>', '', comment)
    comment = [word for word in comment.split(' ') if word not in stopword]
    comment = " ".join(comment)
    comment = [stemmer.stem(word) for word in comment.split(' ')]
    comment = " ".join(comment)
    return comment 

#call clean on entire comment body column: 
#comments_general["commentBody"] = comments_general["commentBody"].apply(clean)
#print(comments_general["commentBody"].head()


#DISPLAY SENTIMENT SCORES ------------

def sentiment_score(x, y, z): 
    total = x+y+z
    if (x>y) and (x>z):
        print("Positive. %d" % (x))
    elif (y>x) and (y>z): 
        print("Negative. %d" % (y))
    else: 
        print("Neutral. %d, %d, %d" % (x, y, z))


#GETTING CATEGORICAL DATA------------------- 

trump_related = articles.keywords.apply(lambda x: 'Trump, Donald J' in x)
articles['trump_related'] = trump_related #column of boolean vals 

trump_df = articles.query('trump_related == True')
#print(len(trump_df))

#277 related articles -- now find the related comments: 
trump_articleIDs = trump_df['articleID'].values.tolist() 
comments_trump = comments.query('articleID in @trump_articleIDs')
"""
#clean them -- comment out other cleaning steps when running this: 
comments_trump["commentBody"] = comments_trump["commentBody"].apply(clean) 
#print(comments_trump["commentBody"].head())

#CALCULATE TRUMP SCORES --------- 
comments_trump["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in 
comments_trump["commentBody"]]

comments_trump["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in 
comments_trump["commentBody"]]

comments_trump["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in 
comments_trump["commentBody"]]

comments_polarity = comments_trump[["commentBody", "Positive", "Negative", "Neutral"]]

#print(comments_polarity.head())

pos = sum(comments_polarity["Positive"])
neg = sum(comments_polarity["Negative"])
neu = sum(comments_polarity["Neutral"])

print("trump related comments scoring: ")
sentiment_score(pos, neg, neu) #this is the analysis for all selected comm. 
"""
#CALCULATE GENERAL SCORES ------ 
comments_general["commentBody"] = comments_general["commentBody"].apply(clean) 
comments_general["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in 
comments_general["commentBody"]]

comments_general["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in 
comments_general["commentBody"]]

comments_general["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in 
comments_general["commentBody"]]

comments_polarity = comments_general[["commentBody", "Positive", "Negative", "Neutral"]]

#print(comments_polarity.head())

pos = sum(comments_polarity["Positive"])
neg = sum(comments_polarity["Negative"])
neu = sum(comments_polarity["Neutral"])

print(" general comments scoring: ")
sentiment_score(pos, neg, neu) #this is the analysis for all selected comm. 
