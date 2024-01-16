#  Import Packages
import numpy as np
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from flask import Flask,jsonify,request 
from flask_cors import CORS
import smtplib


import mysql.connector

import socketio
import time

app = Flask(__name__) 
CORS(app)

mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="semih1306",
            database="tasarim"
        )
 # Kullanıcı id, nabız değerleri frontend'den body kısmı içerisinde gelicek, id değerine göre veritabanında sorgu yapılıp kullanıcının diğer değerlerine erişilecek
@app.route('/measure', methods = ['POST']) 
def measure(): 
    if(request.method == 'POST'):
        veri = request.json
        # Frontend tarafından gelen bilgiler
        mail = veri['mail']
        ad = veri['ad']
        soyad = veri['soyad']
        id = veri['id']
        # Web sitesine giriş yapan kişinin veritabanın'daki id'si üzerinden kullanıcının bilgilerini getiren sorgu
        sorgu2 = f"SELECT * FROM customer WHERE id = {id}"
        mycursor = mydb.cursor()
        mycursor.execute(sorgu2)
        myresult = mycursor.fetchall()
        mylist = list(myresult[0])
        print(mylist)
        # nabiz listenin son elemanı olarak atanıyor
        mylist[-1] = 80
        lastMyList = mylist[5:15]
        print("lastMyList: ",lastMyList)
        lastMyList = [ float(x) for x in lastMyList ]

        
        # Loading dataset
        data = pd.read_csv("data.csv")
        df = data.copy()
        # df = df.reset_index()


        # Model preparation
        ## Features and Labels
        y = df.Heart_attack_risk
        X = df.drop(["Heart_attack_risk"], axis =1).fillna(0).astype("int32")


        ## Feature scaling
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        ## Sınıflandırma modeli
        ## SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        x_ST_cl, y_ST_cl = smote_enn.fit_resample(X, y)
        x_train_smote_enn_KNN, x_test_smote_enn_KNN, y_train_smote_enn_KNN, y_test_smote_enn_KNN = train_test_split(x_ST_cl, y_ST_cl, test_size=0.25, random_state=42)

        hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
        hyper_knn.fit(x_train_smote_enn_KNN,y_train_smote_enn_KNN)
        mylist = np.array(lastMyList)

        tahmin = hyper_knn.predict(mylist.reshape(1,-1))

        print('Sınıflandırma modelinin cıktısı')
        tahminFlag=False
        if tahmin[0] == 1:
            tahminFlag=True
            print(1)
        else:
            tahminFlag=False

        ### Regresyon modeli
        ## SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        x_ST_reg, y_ST_reg = smote_enn.fit_resample(X, y)
        x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST_reg, y_ST_reg, test_size=0.25, random_state=42)


        # Random Forest Regressor modelia
        hOp_forest_model = RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=150)
        hOp_forest_model.fit(x_train_smote_enn,y_train_smote_enn)

        # Test seti üzerinde tahminler yap
        #y_pred_smote_enn = hOp_forest_model.predict(x_test_smote_enn)

        # lastMylist = ['1', '36', '0', '0', '1', '0', '1', '20', '0','90']
        mlist = np.array(lastMyList)


        tahmin2 = hOp_forest_model.predict(mlist.reshape(1,-1))

        print("tahmin:,",tahmin2)
    

        tahminSonuc = tahmin2[0]*100
       
        print('Regresyon modelinin cıktısı')
        print(f"Kalp rahatsızlığı yaşama riskiniz: {tahminSonuc}") 
        # Yapay zeka modeline gönderilecek veriler
 
        # nabiz = veri['nabiz']
        if(tahminSonuc > 50):
            email = "acilsaglikuygulamasi@gmail.com"
            receiver_email = mail

            subject = "Acil Sağlık Bilgilendirmesi"
            message = f"Yakınınız olan {ad} {soyad} kişisinin %{round(tahminSonuc,2)} oranında kalp krizi riski bulunmaktadır"

            text = f"Subject: {subject}\n\n{message}".encode("UTF-8")

            server = smtplib.SMTP("smtp.gmail.com",587)
            server.starttls()

            # kisisel mail
            # server.login(email,"ohiw jjeh xpzl pwex")

            # acil saglik uygulaması maili
            server.login(email,"zwuj zuqh ndmv xkjc")
    

            server.sendmail(email,receiver_email,text)
            print("Email has been sent to " + receiver_email)

        return jsonify(tahminSonuc)


if __name__=='__main__': 
    app.run(debug=True)
