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
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("spam.csv")

df.loc[df["Category"] == 'ham',"Category",] =1
df.loc[df["Category"] == 'spam',"Category",] =0

x= df["Message"]
y=df["Category"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2,train_size=0.8, random_state=4)

tfvec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)#burada tfidvectorizer sınıfından bir nesne oluş
xTrainFeat = tfvec.fit_transform(xTrain)#Bir üst satırda oluşturduğumuz nesneyi Training pandas serisine uydurduk pandas serileri tek boyutlu numpy dizilerine çok benzer
xTestFeat =tfvec.transform(xTest)

# yTrainToInt
yTrainToInt = yTrain.astype('int')
#Multinomial Naive bayes implementation
classifiermodel = MultinomialNB()
classifiermodel.fit(xTrainFeat,yTrainToInt)
predResultNb = classifiermodel.predict(xTestFeat)

#Knn implementation
classifierModelKnn = KNeighborsClassifier()
classifierModelKnn.fit(xTrainFeat,yTrainToInt)
predResultKnn = classifierModelKnn.predict(xTestFeat)

yTest=yTest.astype('int')
actualY = yTest.as_matrix() 


print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
#Accuracy score using MNB
print("Accuracy Score using MNB: {0:.4f}".format(accuracy_score(actualY, predResultNb)*100))
#FScore MACRO using MNB
print("F Score using MNB:{0: .4f}".format(f1_score(actualY, predResultNb, average='macro')*100))
cmMNb=confusion_matrix(actualY, predResultNb)
#"[True negative  False Positive\nFalse Negative True Positive]"1
print("Confusion matrix using MNB:")
print(cmMNb)

print("~~~~~~~~~~Knn RESULTS~~~~~~~~~~")
#Accuracy score using SVM
print("Accuracy Score using Knn: {0:.4f}".format(accuracy_score(actualY, predResultKnn)*100))
#FScore MACRO using SVM
print("F Score using Knn: {0: .4f}".format(f1_score(actualY, predResultKnn, average='macro')*100))
cmSVM=confusion_matrix(actualY, predResultKnn)
#"[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using Knn:")
print(cmSVM)