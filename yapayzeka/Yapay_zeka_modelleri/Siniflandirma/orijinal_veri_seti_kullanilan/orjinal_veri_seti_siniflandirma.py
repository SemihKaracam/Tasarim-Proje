

## Orijinal veri seti (framingham.csv) ön işleme sonrası data0.csv dosyası olarak kaydedildi.
## Orijinal veri seti (data0.csv) kullanılarak Sınıflandırma modelleri geliştirildi.
## Sınıflandırma modellerinin performansları karşılaştırıldı.


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from collections import Counter

from sklearn.preprocessing import LabelEncoder  #preprocessing
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split    #splitting data

from sklearn.model_selection import cross_val_score,KFold   #model selection


from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


data = pd.read_csv("framingham.csv")
df = data.copy()

# Sütun adını değiştirelim
df = df.rename(columns={'male': 'gender', 'diabetes':'prevalentDiabetes','TenYearCHD':'Heart_attack_risk'})


## Verinin İncelenmesi ###########################################

# NAN değerlerin doldurulması

# NaN değerlerin bulunduğu sütunları bulma
nan_sutunlar = df.columns[df.isna().any()].tolist()

# # NaN değerlerini sütunların ortalamasıyla doldurma
for sutun in nan_sutunlar:
    ort = df[sutun].mean()
    df[sutun].fillna(ort, inplace=True)

df.to_csv('data0.csv', index=False) # orijinal veri setinin ön işleme sonrası hali

## Üzerinde çalışacağımız veri seti
data = pd.read_csv("data0.csv")
df = data.copy()

# print(df.head())
# print(df.info())
# print(df.describe())

## Verinin Görselleştirilmesi ###########################################

# Isı haritası
corr = df.corr()
fig, ax = plt.subplots(figsize = (18,12))
ax = sns.heatmap(corr, annot = True , fmt ='.2f')

# Veri setinin dağılımı
df.hist(bins=30, 
        figsize=(20,40),
        layout=(15,4))

# Özellik önem siralaması
rf = RandomForestClassifier()
rf.fit( df.iloc[:,:-1], df.iloc[:,-1] )
importance=rf.feature_importances_
df3=pd.DataFrame({"Features":df.iloc[:,:-1].columns,"Importance":importance})
sns.barplot(x="Features",y="Importance",data=df3,order=df3.sort_values("Importance")['Features'])
plt.xticks(rotation=90)
plt.show()



## Veri setini Makine Öğrenimi Modellerine Uydurma #########################

# # Veri setindeki sınıf dengesizliğini  dengeleme


## Özellikler ve etiketler
y = df.Heart_attack_risk
X = df.drop(["Heart_attack_risk"], axis =1).astype("float64")

## Normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

## Eğitim ve test veri setinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

## Modellerin oluşturulması
log_reg = LogisticRegression()  #Logistic Regression
naive_bayes = GaussianNB()      #Naive Bayes
rand_forest = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)  #Random Forest Classifier
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
knn = KNeighborsClassifier(n_neighbors=9) #K-Neighbors Classifier
dec_tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)   #Decision Tree Classifier
svm = SVC(C=2, kernel='rbf')   #Support Vector Machine

## Modellerin eğitilmesi
log_reg_model = log_reg.fit(X_train,y_train)
naive_bayes_model = naive_bayes.fit(X_train,y_train)
rand_forest_model = rand_forest.fit(X_train,y_train)
xgb_model = xgb.fit(X_train,y_train)
knn_model = knn.fit(X_train, y_train)
dec_tree_model = dec_tree.fit(X_train, y_train)
svm_model = svm.fit(X_train, y_train)

## Modellerin tahmin yapması
log_reg_predict = log_reg.predict(X_test)
naive_bayes_predict = naive_bayes.predict(X_test)
rand_forest_predict = rand_forest.predict(X_test)
xgb_predict =xgb.predict(X_test)
knn_predict= knn.predict(X_test)
dec_tree_predict = dec_tree.predict(X_test)
svm_predict = svm.predict(X_test)

## Model performanslarının karşılaştırılması

# Doğruluk
# print("Accuracy of Logistic Regression:", accuracy_score(y_test, log_reg_predict))
# print("Accuracy of Naive Bayes:", accuracy_score(y_test, naive_bayes_predict))
# print("Accuracy of Random Forest Classifier:", accuracy_score(y_test, rand_forest_predict))
# print("Accuracy of Extreme Gradient Boost:", accuracy_score(y_test, xgb_predict))
# print("Accuracy of K-Neighbors Classifier:", accuracy_score(y_test, knn_predict))
# print("Accuracy of Decision Tree Classifier:", accuracy_score(y_test, dec_tree_predict))
# print("Accuracy of Support Vector Classifier:", accuracy_score(y_test, svm_predict))

# Karmaşıklık matrisi
print("Confussion matrix of Logistic Regression:")
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, log_reg_predict), annot=True, fmt="d", cmap="Blues")
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()


# Sınıflandırma raporu
print("Classification report of Logistic Regression:\n", classification_report(y_test, log_reg_predict) )
print("Classification report of Naive Bayes:\n", classification_report(y_test, naive_bayes_predict) )
print("Classification report of Random Forest Classifier:\n", classification_report(y_test, rand_forest_predict) )
print("Classification report of Extreme Gradient Boost:\n", classification_report(y_test, xgb_predict) )
print("Classification report of K-Neighbors Classifier:\n", classification_report(y_test, knn_predict) )
print("Classification report of Decision Tree Classifier:\n", classification_report(y_test, dec_tree_predict) )
print("Classification report of Support Vector Classifier:\n", classification_report(y_test, svm_predict) )

# ROC eğrisi
lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,log_reg_predict)
nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,naive_bayes_predict)
rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,rand_forest_predict)                                                             
xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predict)
knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,knn_predict)
dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,dec_tree_predict)
svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,svm_predict)


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('ROC Eğrisi')
plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive degeri')
plt.xlabel('False positive degeri')
plt.legend()
plt.show()

