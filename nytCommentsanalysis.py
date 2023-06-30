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

#get articleID lists for each category: 
politics = art_df.query("newDesk in @lst_topics[0]")
culture = art_df.query("newDesk in @lst_topics[1]")
national = art_df.query("newDesk in @lst_topics[2]")

#READING IN COMMENT DETAILS ------------

#using articleID, commentBody, and commentID:
comments = pd.read_csv("/Users/sharrutatachar/Desktop/CommentsApril2018.csv",
usecols = list(range(1,2)) + list(range(3,5)))

#get the comments that are relevant to the chosen articles: 
articleIDs = art_df['articleID'].values.tolist() 
comments = comments.query('articleID in @articleIDs') #around 24k comments


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
comments["commentBody"] = comments["commentBody"].apply(clean)
#print(comments["commentBody"].head())


#CALCULATE OVERALL SENTIMENT SCORES ------------

comments["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in 
comments["commentBody"]]

comments["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in 
comments["commentBody"]]

comments["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in 
comments["commentBody"]]

comments_polarity = comments[["commentBody", "Positive", "Negative", "Neutral"]]

#print(comments_polarity.head())

pos = sum(comments_polarity["Positive"])
neg = sum(comments_polarity["Negative"])
neu = sum(comments_polarity["Neutral"])

def sentiment_score(x, y, z): 
    if (x>y) and (x>z):
        print("Positive overall. %d" % (x))
    elif (y>x) and (y>z): 
        print("Negative overall. %d" % (y))
    else: 
        print("Neutral overall. %d, %d, and %d" % (x, y, z))

sentiment_score(pos, neg, neu) #this is the analysis for all selected comm. 

#TO-DO: CALCULATE SENTIMENT SCORES FOR EACH CATEGORY AND COMPARE---------