import pandas as pd 
import numpy as np 
#from sklearn.tree import DecisionTreeClassifier 
import re 
import string 
import nltk 
#nltk stuff: 
nltk.download('stopwords')
stemmer = nltk.SnowbalStemmer("english")
stopword = set(stopwords.words('english'))


#READING IN ARTICLE DETAILS ------------

#only using articleID, headlines, newDesk, keywords:
articles = pd.read_csv("/Users/sharrutatachar/Desktop/ArticlesApril2018.csv", 
usecols = list(range(0,1)) + list(range(4,8)))
del articles['multimedia']
#print(articles.head())

#READING IN COMMENT DETAILS ------------

#using articleID, commentBody, and commentID:
comments = pd.read_csv("/Users/sharrutatachar/Desktop/CommentsApril2018.csv",
usecols = list(range(1,2)) + list(range(3,5)))
#print(comments.head())


#CLEAN COMMENT BODY ------------

def clean(comment): 
    comment = str(comment).lower()
    comment = re.sub('<br/><br/>', '', comment)
    comment = [word for word in comment.split(' ') if word not in stopword]
    comment = " ".join(comment)
    coment = [stemmer.stem(word) for word in comment.split(' ')]
    comment = " ".join(comment)
    return comment 

#call clean on entire comment body column: 
comments["commentBody"] = comments["commentBody"].apply(clean)

