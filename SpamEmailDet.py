# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:57:19 2019

@author: ASUS
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

df = pd.read_csv("spam.csv")

df.loc[df["Category"] == 'ham',"Category",] =1
df.loc[df["Category"] == 'spam',"Category",] =0

x= df["Message"]
y=df["Category"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2,train_size=0.8, random_state=4)

tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)#burada tfidvectorizer sınıfından bir nesne oluş
xTrainFeat = tfvec.fit_transform(xTrain)#Bir üst satırda oluşturduğumuz nesneyi Training pandas serisine uydurduk pandas serileri tek boyutlu numpy dizilerine çok benzer
xTestFeat =tfvec.transform(xTest)

