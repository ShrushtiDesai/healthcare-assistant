from ast import Str
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as LabelEncoder
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder as LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import mode
import pyttsx3
import datetime
import time
import playsound
import speech_recognition as sr
from gtts import gTTS

def model():
    train = pd.read_csv(r"C:\Users\Shrushti\Downloads\Training.csv\Training.csv")
    test = pd.read_csv(r"C:\Users\Shrushti\Downloads\Testing.csv")
    A = train
    B = test
# assigning A and B to train and test is not that useful
# Encoding the target value into numerical
# value using LabelEncoder
    encoder = LabelEncoder()
    A["prognosis"] = encoder.fit_transform(A["prognosis"])

    A = A.drop(["Unnamed: 133"],axis=1)

    Y = A[["prognosis"]]
    X = A.drop(["prognosis"],axis=1)
    P = B.drop(["prognosis"],axis=1)
#TRAINING IS SPLIT INOT TWO PARTS 
    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)
# ############################################################################################################################################################
    dtc= DecisionTreeClassifier(random_state=42)
    dtc_model = dtc.fit(xtrain,ytrain)
    tr_pred_dtc = dtc_model.predict(xtrain)
    ts_pred_dtc = dtc_model.predict(xtest)

# print("Training data accuracy of Decision Tree Classifier is : " +str(accuracy_score(ytrain,tr_pred_dtc)*100)+ " %")
# print("Testing data accuracy of Decision Tree Classifier is : " +str(accuracy_score(ytest,ts_pred_dtc)*100) + " %\n")

# Training and testing Naive Bayes Classifier
    nbc = GaussianNB()
    nbc_model = nbc.fit(xtrain, ytrain)
    tr_pred_nbc = nbc_model.predict(xtrain)
    ts_pred_nbc = nbc_model.predict(xtest)

# print("Training data accuracy of Naive Bayes Classifier is : " +str(accuracy_score(ytrain, tr_pred_nbc)*100)+" %")
# print("Testing data accuracy of Naive Bayes Classifier is : " +str(accuracy_score(ytest, ts_pred_nbc)*100)+" %\n")
# cf_matrix = confusion_matrix(ytest, ts_pred_nbc)
# plt.figure(figsize=(12,8))
# sns.heatmap(cf_matrix, annot=True)                                                    #plotting is not neccessary can discard
# plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
# plt.show()
#prediction for full train dataset which was previously splited into two and then it will be compared with the test dataset 
    dtc_final_model = DecisionTreeClassifier()
    nbc_final_model = GaussianNB()
    dtc_final_model.fit(X,Y)
    nbc_final_model.fit(X,Y)

    test_X = B.iloc[:, :-1]
    test_Y = encoder.transform(B.iloc[:, -1])

    dtc_final_pred = dtc_final_model.predict(test_X)
    nbc_final_pred = nbc_final_model.predict(test_X)

    Final_prediction = [mode([i,j])[0][0] for i,j
                         in zip(dtc_final_pred,nbc_final_pred)]

# print("Test data combined accuracy of both the models is : " +str(accuracy_score(test_Y, Final_prediction)*100)+" %")
#combine data accuracy is getting reduced find why????
