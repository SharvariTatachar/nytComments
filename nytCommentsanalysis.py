import pandas as pd 
import numpy as np 
import re 
import string 
from nltk.corpus import stopwords
import nltk
stemmer = nltk.SnowballStemmer("english")
stopword = stopwords.words('english')


#READING IN ARTICLE DETAILS ------------

#only using articleID, headlines, newDesk, keywords:
articles = pd.read_csv("/Users/sharrutatachar/Desktop/ArticlesApril2018.csv", 
usecols = list(range(0,1)) + list(range(4,8)))
del articles['multimedia']
#articles = articles[0:663] #truncate dataset for smaller set  
lst_topics = ["Politics", "Culture", "National"]
art_df = articles.query('newDesk in @lst_topics') 


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
print(comments["commentBody"].head())
