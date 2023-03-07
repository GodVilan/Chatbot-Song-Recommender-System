from spacy.lang.en import English
import numpy
from flask import Flask, render_template, request
import json
import pickle
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from voc import voc
import random
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
PAD_Token=0
app = Flask(__name__)
model= models.load_model('/mymodel.h5')
with open("mydata.pickle", "rb") as f:
    data = pickle.load(f)
# Libraries
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
#pip install librosa==0.7.2
#pip install numba==0.49.1
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pymysql
@app.route("/upload")
def upload():
    return render_template('/home.html')
@app.route('/home', methods = ['GET', 'POST'])
def home1():
    if request.method == 'POST' and request.files['myfile']:
        myfile = request.files['myfile']
        file1=myfile.filename
        file="/speech recognitionemotion/Actor_10/"+file1
        feature=extract_feature(file,mfcc=True, chroma=True, mel=True)
        x_train,x_test,y_train,y_test=load_data(test_size=0.5)
        print((x_train.shape[0], x_test.shape[0]))
        print(f'Features extracted: {x_train.shape[1]}')
        model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
        hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=800)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        ypre=model.predict([feature])
        accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy: {:.2f}%".format(accuracy*100))
        print(ypre)
        d={'result':ypre,'accuracy':accuracy*100}
        import webbrowser
        #url = 'http://docs.python.org/'
        chrome_path = 'C:/Program Files(x86)/Google/Chrome/Application/chrome.exe %s'
        #webbrowser.get(chrome_path).open(url)
        #time.sleep(2)
        if ypre[0]=="calm":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        elif ypre[0]=="nuetral":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        elif ypre[0]=="happy":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        elif ypre[0]=="sad":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        elif ypre[0]=="angry":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        elif ypre[0]=="fearful":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        elif ypre[0]=="disgust":
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        else:
            webbrowser.get(chrome_path).open("https://www.youtube.com/watch?v=iv7lcUkFVSc")
        return render_template('result.html',data=d)
emotions={
'01':'neutral',
'02':'calm',
'03':'happy',
'04':'sad',
'05':'angry',
'06':'fearful',
'07':'disgust',
'08':'surprised'
}
observed_emotions=['calm', 'happy', 'fearful', 'disgust']
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("/speech recognitionemotion/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
    if chroma:
        stft=np.abs(librosa.stft(X))
        result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
        n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft,
        sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X,
        sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

def predict(ques):
    ques= data.getQuestionInNum(ques)
    ques=numpy.array(ques)
    # ques=ques/255
    ques = numpy.expand_dims(ques, axis = 0)
    y_pred = model.predict(ques)
    res=numpy.argmax(y_pred, axis=1)
    return res

def getresponse(results):
    tag= data.index2tags[int(results)]
    response= data.response[tag]
    return response
def chat(inp):
    while True:
        inp_x=inp.lower()
        results = predict(inp_x)
        response= getresponse(results)
    return random.choice(response)
@app.route("/")
def home():
    return render_template("/index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    time.sleep(1)
    return str(chat(userText))
if __name__ == "__main__":
    app.run()