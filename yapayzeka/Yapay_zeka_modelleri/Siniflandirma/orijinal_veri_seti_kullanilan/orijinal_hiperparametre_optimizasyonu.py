

## Orijinal veri seti kullanılarak geliştirilecek olan sınıflandırma modellerine hiper parametre optimizasyonu yapıldı.
## Sınıflandırma modellerinin performansları karşılaştırıldı.


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from collections import Counter

from sklearn.preprocessing import LabelEncoder  #preprocessing
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split    #splitting data

#from sklearn.model_selection import cross_val_score,KFold   #model selection

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report, auc


# Orijinal veri seti icin
df = pd.read_csv("data0.csv")

# df.head()
# df.info()

# df.describe()

################################## 

y = df.Heart_attack_risk
X = df.drop(["Heart_attack_risk"], axis =1).astype("float64")

#normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


####################################### Hiper parametre optimizasyonu ##############################

##################### LOGİSTİC REGRESSION ############################

# log_reg = LogisticRegression()  #Logistic Regression

# hyp_param_LR = {
#     "penalty": ["l1", "l2"],
#     "C": list(range(1, 5)),
#     "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
#     "max_iter": list(range(100, 600, 100)),
#     "class_weight": [{0: 0.15, 1: 0.85}, {0: 0.13, 1: 0.75}, {0: 0.1, 1: 0.95}, {0: 1, 1: 1}]
# }

# classfier_LR = GridSearchCV(log_reg , hyp_param_LR , cv=5)
# classfier_LR.fit(X_train,y_train)

# # print(classfier_LR.best_params_)   #{'C': 4, 'class_weight': {0: 1, 1: 1}, 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}
# hyper_log_reg = LogisticRegression(C = 4, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})

# hyper_log_reg.fit(X_train , y_train)
# hyper_log_reg_predict = hyper_log_reg.predict(X_test)

# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test, hyper_log_reg_predict) )


# ## Karmaşıklık matrisi
# print("Confussion matrix of Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test, hyper_log_reg_predict), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()

##################### NAIVE BAYES ############################

# naive_bayes = GaussianNB()     # Naive bayes

# hyp_param_NB = {
#     "priors": [None] , # priors: Sınıf prior olasılıkları
#     "var_smoothing": [1e-9, 1e-8, 1e-7]  # var_smoothing: Varyans düzeltme
# }

# classfier_NB = GridSearchCV(naive_bayes , hyp_param_NB , cv=5)
# classfier_NB.fit(X_train,y_train)

# print(classfier_NB.best_params_)   #{'priors': None, 'var_smoothing': 1e-09}#
# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)

# hyper_NB.fit(X_train , y_train)
# hyper_NB_predict = hyper_NB.predict(X_test)

# # print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test, hyper_NB_predict) )

# ##################### RANDOM FOREST ############################

# random_forest = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)  #Random Forest Classifier  

# hyp_param_RF = {  # n_estimators: Oluşturulacak ağaç sayısı
#     "criterion": ["gini", "entropy"],  # criterion: Bölme kriteri
#     "max_depth": [None, 10, 20, 30],  # max_depth: Ağaçların maksimum derinliği
#     "min_samples_split": [2, 5, 10],  # min_samples_split: Bir iç düğüm bölünmeden önce gereken minimum örnek sayısı
#     "min_samples_leaf": [1, 2, 4],  # min_samples_leaf: Bir yaprak düğümde gereken minimum örnek sayısı
#     "max_features": ["auto", "sqrt", "log2"],  # max_features: Her bölme için göz önüne alınacak maksimum özellik sayısı
#     "class_weight": [None, "balanced"]  # class_weight: Sınıf ağırlıkları
# }

# classfier_RF = GridSearchCV(random_forest , hyp_param_RF , cv=5)
# classfier_RF.fit(X_train,y_train)

# # print(classfier_RF.best_params_) # {'class_weight': None, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10}
# hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=1, min_samples_split=10) 

# hyper_RF.fit(X_train , y_train)
# hyper_RF_predict = hyper_RF.predict(X_test)

# print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test, hyper_RF_predict) )


# # # ##################### EXTREME GRADIENT BOOST ############################

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                     reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost

# hyp_param_XGB = {
#     "learning_rate": [0.01, 0.1, 0.2],  # learning_rate: Öğrenme oranı, her ağacın katkısının boyutunu kontrol eder
#     "n_estimators": [50, 100, 200],  # n_estimators: Oluşturulan ağaç sayısı
#     "max_depth": [3, 5, 7],  # max_depth: Her ağacın maksimum derinliği
#     "min_child_weight": [1, 3, 5],  # min_child_weight: Bir iç düğümde gereken minimum örnek ağırlığı
#     "colsample_bytree": [0.8, 0.9, 1.0],  # colsample_bytree: Her ağaç için kullanılacak özellik oranı
#     "gamma": [0, 0.1, 0.2],  # gamma: Bir düğümün bölünmesinin anlamlılığını kontrol eden bir parametre
#     "reg_alpha": [0, 0.1, 0.2],  # reg_alpha: L1 düzenleme terimi
#     "reg_lambda": [1, 1.5, 2],  # reg_lambda: L2 düzenleme terimi
# }


# classfier_XGB = GridSearchCV(xgb , hyp_param_XGB , cv=5)
# print('bitti')
# classfier_XGB.fit(X_train,y_train)

# print(classfier_XGB.best_params_)   

# # hyper_XGB = XGBClassifier()

# # hyper_XGB.fit(X_train , y_train)
# # hyper_XGB_predict = hyper_XGB.predict(X_test)

# # # print("Classification report of Hyper optimized XGB:\n", classification_report(y_test, hyper_XGB_predict) )



# # ##################### KNN ############################

# knn = KNeighborsClassifier(n_neighbors=9) # KNN

# hyp_param_KNN = {
#     "n_neighbors": [3, 5, 7],
#     "weights": ["uniform", "distance"],
#     "p": [1, 2]  # p: Minkowski mesafe metrik parametresi
# }

# classfier_KNN = GridSearchCV(knn , hyp_param_KNN , cv=5)
# classfier_KNN.fit(X_train,y_train)

# print(classfier_KNN.best_params_)  #{'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}
# hyper_KNN = KNeighborsClassifier(n_neighbors=7, p=1, weights='uniform')

# hyper_KNN.fit(X_train , y_train)
# hyper_KNN_predict = hyper_KNN.predict(X_test)

# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test, hyper_KNN_predict) )


# # ##################### DECISION TREE ############################

# dec_tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)   #Decision Tree Classifier

# hyp_param_DT = {
#     "criterion": ["gini", "entropy"],
#     "max_depth": [3, 5, 7],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["auto", "sqrt", "log2"]
# }

# classfier_DT = GridSearchCV(dec_tree , hyp_param_DT , cv=5)
# classfier_DT.fit(X_train,y_train)

# # print(classfier_DT.best_params_) #{'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 10}
# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10)

# hyper_DT.fit(X_train , y_train)
# hyper_DT_predict = hyper_DT.predict(X_test)

# # print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test, hyper_DT_predict) )


# # ##################### SUPPORT VECTOR MACHINE ############################

# svm = SVC(C=2, kernel='rbf')   #Support Vector Machine

# hyp_param_SVC = {
#     "C": [0.1, 1, 10],
#     "kernel": ["linear", "poly", "rbf", "sigmoid"],
#     "gamma": ["scale", "auto", 0.1, 1],
#     "degree": [2, 3, 4]  
# }

# classfier_SVC = GridSearchCV(svm , hyp_param_SVC , cv=5)

# classfier_SVC.fit(X_train,y_train)

# print(classfier_SVC.best_params_)   #{'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 

# hyper_SVC.fit(X_train , y_train)
# hyper_SVC_predict = hyper_SVC.predict(X_test)

# print("Classification report of Hyper optimized Support Vector Classifier:\n", classification_report(y_test, hyper_SVC_predict) )


# lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test,hyper_log_reg_predict)
# nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test,hyper_NB_predict)
# rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test,hyper_RF_predict)                                                             
# # xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predict)
# knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test,hyper_KNN_predict)
# dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test,hyper_DT_predict)
# svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test,hyper_SVC_predict)


# sns.set_style('whitegrid')
# plt.figure(figsize=(10,5))
# plt.title('ROC Eğrisi')
# plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
# plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
# plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
# # plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
# plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
# plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
# plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
# plt.plot([0,1],ls='--')
# plt.plot([0,0],[1,0],c='.5')
# plt.plot([1,1],c='.5')
# plt.ylabel('True positive degeri')
# plt.xlabel('False positive degeri')
# plt.legend()
# plt.show()