

## 7 farklı veri seti dengeleme yontemi kullanılarak veri seti dengelendi.
## 9 farklı regresyon modeli eğitildi.

## Dengeleme yöntemleri ve regresyon modellerinin performansları karşılaştırıldı.
## En iyi sonuca SMOTEENyöntemi ve Rasgele Orman Regresyonu modeliyle ulasıldı.


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder  #preprocessing
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split    #splitting data
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report, auc

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score


## Techniques to Convert Imbalanced Dataset into Balanced Dataset With Random Forest Classifier¶
## 1) Random Oversampling with Evaluation
## 1.1) With SMOTE (Synthetic Minority Oversampling Technique) with Evaluation
## 1.2) ADASYN Random over-sampling Evaluation
## 2) Random Under-Sampling With Evaluation
## 2.1) With Near Miss Under-Sampling With Evaluation
## 3) Ensemble Learning Techniques with Evaluation
## 3.1 ) Bagging classifier
## 3.2) Boosting
## 3.3) Forest of randomized trees
## 4) Combination of over- and under-sampling With Evaluation
## 4.1) SMOTEENN
## 4.2) SMOTETomek

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE    #pip3 install imblearn
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


# Duzenlenmis veri seti icin
data = pd.read_csv("data.csv")
df = data.copy()

# df.head()
# df.info()
# df.describe()

y = df.Heart_attack_risk
X = df.drop(["Heart_attack_risk"], axis =1).astype("float64")

#normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
 

##################### LİNEER REGRESSION #############################################################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# LR = LinearRegression()
# LR.fit(x_train_ros,y_train_ros)

# y_pred_ros = LR.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')

# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# #Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))


# # SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# LR = LinearRegression()
# LR.fit(x_train_smote,y_train_smote)

# y_pred_smote = LR.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')

# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# #Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))



# # ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# LR = LinearRegression()
# LR.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = LR.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')

# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# #Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# LR = LinearRegression()
# LR.fit(x_train_rus,y_train_rus)

# y_pred_rus = LR.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')

# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# #Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# LR = LinearRegression()
# LR.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = LR.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# #Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



# SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

LR = LinearRegression()
LR.fit(x_train_smote_enn,y_train_smote_enn)

y_pred_smote_enn = LR.predict(x_test_smote_enn)
print('\n\nEvaluation with SMOTEENN and LinearRegression:')

mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# LR = LinearRegression()
# LR.fit(x_train_smote_tomek,y_train_smote_tomek)
# y_pred_smote_tomek = LR.predict(x_test_smote_tomek)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')


# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))







# ##################### Polinominal Regresyon #############################################################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_ros)
# X_test_poly = poly_features.transform(x_test_ros)

# # Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
# model = LinearRegression()
# model.fit(X_train_poly, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = model.predict(X_test_poly)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))



# # SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote)
# X_test_poly = poly_features.transform(x_test_smote)

# # Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
# model = LinearRegression()
# model.fit(X_train_poly, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))



# # ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_adasyn)
# X_test_poly = poly_features.transform(x_test_adasyn)

# # Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
# model = LinearRegression()
# model.fit(X_train_poly, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = model.predict(X_test_poly)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_rus)
# X_test_poly = poly_features.transform(x_test_rus)

# # Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
# model = LinearRegression()
# model.fit(X_train_poly, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = model.predict(X_test_poly)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_nearMiss)
# X_test_poly = poly_features.transform(x_test_nearMiss)

# # Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
# model = LinearRegression()
# model.fit(X_train_poly, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = model.predict(X_test_poly)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



# SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# PolynomialFeatures'ı kullanarak özellikleri genişlet
degree = 2  # Genişletme derecesi
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(x_train_smote_enn)
X_test_poly = poly_features.transform(x_test_smote_enn)

# Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
model = LinearRegression()
model.fit(X_train_poly, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = model.predict(X_test_poly)

print('\n\nEvaluation with SMOTEENN and Polinominal Regresyon:')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote_tomek)
# X_test_poly = poly_features.transform(x_test_smote_tomek)

# # Genişletilmiş özellik seti üzerinde Lineer Regresyon modelini oluştur
# model = LinearRegression()
# model.fit(X_train_poly, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))


##################### Ridge Regresyon #############################################################################################################


# ## RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_ros)
# X_test_poly = poly_features.transform(x_test_ros)

# # Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
# alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_poly, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = ridge_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))




# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote)
# X_test_poly = poly_features.transform(x_test_smote)

# # Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
# alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_poly, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = ridge_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_adasyn)
# X_test_poly = poly_features.transform(x_test_adasyn)

# # Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
# alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_poly, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = ridge_model.predict(X_test_poly)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_rus)
# X_test_poly = poly_features.transform(x_test_rus)

# # Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
# alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_poly, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = ridge_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_nearMiss)
# X_test_poly = poly_features.transform(x_test_nearMiss)

# # Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
# alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_poly, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = ridge_model.predict(X_test_poly)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))




## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# PolynomialFeatures'ı kullanarak özellikleri genişlet
degree = 2  # Genişletme derecesi
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(x_train_smote_enn)
X_test_poly = poly_features.transform(x_test_smote_enn)

# Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_poly, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = ridge_model.predict(X_test_poly)

print('\n\nEvaluation with SMOTEENN and Ridge Regresyon:')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote_tomek)
# X_test_poly = poly_features.transform(x_test_smote_tomek)

# # Genişletilmiş özellik seti üzerinde Ridge Regresyon modelini oluştur
# alpha = 1.0  # Ridge regresyon katsayısı (alpha) - Düzenleme parametresi
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_poly, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = ridge_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))



# ############################## Lasso Regresyon ##############################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_ros)
# X_test_poly = poly_features.transform(x_test_ros)

# # Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
# alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
# lasso_model = Lasso(alpha=alpha)
# lasso_model.fit(X_train_poly, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = lasso_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote)
# X_test_poly = poly_features.transform(x_test_smote)

# # Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
# alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
# lasso_model = Lasso(alpha=alpha)
# lasso_model.fit(X_train_poly, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = lasso_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))




# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_adasyn)
# X_test_poly = poly_features.transform(x_test_adasyn)

# # Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
# alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
# lasso_model = Lasso(alpha=alpha)
# lasso_model.fit(X_train_poly, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = lasso_model.predict(X_test_poly)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_rus)
# X_test_poly = poly_features.transform(x_test_rus)

# # Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
# alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
# lasso_model = Lasso(alpha=alpha)
# lasso_model.fit(X_train_poly, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = lasso_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_nearMiss)
# X_test_poly = poly_features.transform(x_test_nearMiss)

# # Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
# alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
# lasso_model = Lasso(alpha=alpha)
# lasso_model.fit(X_train_poly, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = lasso_model.predict(X_test_poly)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# PolynomialFeatures'ı kullanarak özellikleri genişlet
degree = 2  # Genişletme derecesi
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(x_train_smote_enn)
X_test_poly = poly_features.transform(x_test_smote_enn)

# Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_train_poly, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = lasso_model.predict(X_test_poly)

print('\n\nEvaluation with SMOTEENN and Lasso Regresyon:')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote_tomek)
# X_test_poly = poly_features.transform(x_test_smote_tomek)

# # Genişletilmiş özellik seti üzerinde Lasso Regresyon modelini oluştur
# alpha = 1.0  # Lasso regresyon katsayısı (alpha) - Düzenleme parametresi
# lasso_model = Lasso(alpha=alpha)
# lasso_model.fit(X_train_poly, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = lasso_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))



######################################## Elastik Net Regresyon ########################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_ros)
# X_test_poly = poly_features.transform(x_test_ros)

# # Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
# alpha = 1.0  # Düzenleme katsayısı
# l1_ratio = 0.5  # L1 oranı
# elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# elasticnet_model.fit(X_train_poly, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = elasticnet_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))




# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote)
# X_test_poly = poly_features.transform(x_test_smote)

# # Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
# alpha = 1.0  # Düzenleme katsayısı
# l1_ratio = 0.5  # L1 oranı
# elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# elasticnet_model.fit(X_train_poly, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = elasticnet_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))




# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_adasyn)
# X_test_poly = poly_features.transform(x_test_adasyn)

# # Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
# alpha = 1.0  # Düzenleme katsayısı
# l1_ratio = 0.5  # L1 oranı
# elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# elasticnet_model.fit(X_train_poly, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = elasticnet_model.predict(X_test_poly)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_rus)
# X_test_poly = poly_features.transform(x_test_rus)

# # Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
# alpha = 1.0  # Düzenleme katsayısı
# l1_ratio = 0.5  # L1 oranı
# elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# elasticnet_model.fit(X_train_poly, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = elasticnet_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_nearMiss)
# X_test_poly = poly_features.transform(x_test_nearMiss)

# # Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
# alpha = 1.0  # Düzenleme katsayısı
# l1_ratio = 0.5  # L1 oranı
# elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# elasticnet_model.fit(X_train_poly, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = elasticnet_model.predict(X_test_poly)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# PolynomialFeatures'ı kullanarak özellikleri genişlet
degree = 2  # Genişletme derecesi
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(x_train_smote_enn)
X_test_poly = poly_features.transform(x_test_smote_enn)

# Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
alpha = 1.0  # Düzenleme katsayısı
l1_ratio = 0.5  # L1 oranı
elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
elasticnet_model.fit(X_train_poly, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = elasticnet_model.predict(X_test_poly)

print('\n\nEvaluation with SMOTEENN and Elastik Net Regresyon:')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')


# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')


# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote_tomek)
# X_test_poly = poly_features.transform(x_test_smote_tomek)

# # Genişletilmiş özellik seti üzerinde ElasticNet Regresyon modelini oluştur
# alpha = 1.0  # Düzenleme katsayısı
# l1_ratio = 0.5  # L1 oranı
# elasticnet_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
# elasticnet_model.fit(X_train_poly, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = elasticnet_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))




######################################## Support Vector Regression (SVR) #######################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_ros)
# X_test_poly = poly_features.transform(x_test_ros)

# # Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
# svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
# svr_model.fit(X_train_poly, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = svr_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))




# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote)
# X_test_poly = poly_features.transform(x_test_smote)

# # Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
# svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
# svr_model.fit(X_train_poly, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = svr_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))




# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_adasyn)
# X_test_poly = poly_features.transform(x_test_adasyn)

# # Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
# svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
# svr_model.fit(X_train_poly, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = svr_model.predict(X_test_poly)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_rus)
# X_test_poly = poly_features.transform(x_test_rus)

# # Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
# svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
# svr_model.fit(X_train_poly, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = svr_model.predict(X_test_poly)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_nearMiss)
# X_test_poly = poly_features.transform(x_test_nearMiss)

# # Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
# svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
# svr_model.fit(X_train_poly, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = svr_model.predict(X_test_poly)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# PolynomialFeatures'ı kullanarak özellikleri genişlet
degree = 2  # Genişletme derecesi
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly_features.fit_transform(x_train_smote_enn)
X_test_poly = poly_features.transform(x_test_smote_enn)

# Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
svr_model.fit(X_train_poly, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = svr_model.predict(X_test_poly)

print('\n\nEvaluation with SMOTEENN and Support Vector Regression (SVR):')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')


# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))


# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)


# # PolynomialFeatures'ı kullanarak özellikleri genişlet
# degree = 2  # Genişletme derecesi
# poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# X_train_poly = poly_features.fit_transform(x_train_smote_tomek)
# X_test_poly = poly_features.transform(x_test_smote_tomek)

# # Genişletilmiş özellik seti üzerinde Support Vector Regression modelini oluştur
# svr_model = SVR(kernel='linear')  # 'linear' kernel kullanılabilir, veya farklı kernel seçenekleri de denenebilir
# svr_model.fit(X_train_poly, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = svr_model.predict(X_test_poly)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))




########################################### Decision Trees Regression #############################################################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# # DecisionTreeRegressor modelini oluştur
# dt_model = DecisionTreeRegressor()
# dt_model.fit(x_train_ros, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = dt_model.predict(x_test_ros)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))




# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# # DecisionTreeRegressor modelini oluştur
# dt_model = DecisionTreeRegressor()
# dt_model.fit(x_train_smote, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = dt_model.predict(x_test_smote)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# # DecisionTreeRegressor modelini oluştur
# dt_model = DecisionTreeRegressor()
# dt_model.fit(x_train_adasyn, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = dt_model.predict(x_test_adasyn)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# # DecisionTreeRegressor modelini oluştur
# dt_model = DecisionTreeRegressor()
# dt_model.fit(x_train_rus, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = dt_model.predict(x_test_rus)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# # DecisionTreeRegressor modelini oluştur
# dt_model = DecisionTreeRegressor()
# dt_model.fit(x_train_nearMiss, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = dt_model.predict(x_test_nearMiss)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# DecisionTreeRegressor modelini oluştur
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train_smote_enn, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = dt_model.predict(x_test_smote_enn)

print('\n\nEvaluation with SMOTEENN and Decision Trees Regression:')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')


# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))




# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# # DecisionTreeRegressor modelini oluştur
# dt_model = DecisionTreeRegressor()
# dt_model.fit(x_train_smote_tomek, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = dt_model.predict(x_test_smote_tomek)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))


############################## Random Forest Regression ####################################################################33

# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# # RandomForestRegressor modelini oluştur
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(x_train_ros, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = rf_model.predict(x_test_ros)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))




# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# # RandomForestRegressor modelini oluştur
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(x_train_smote, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote = rf_model.predict(x_test_smote)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yaoszdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# # RandomForestRegressor modelini oluştur
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(x_train_adasyn, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = rf_model.predict(x_test_adasyn)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# # RandomForestRegressor modelini oluştur
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(x_train_rus, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = rf_model.predict(x_test_rus)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# # RandomForestRegressor modelini oluştur
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(x_train_nearMiss, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = rf_model.predict(x_test_nearMiss)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# RandomForestRegressor modelini oluştur
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train_smote_enn, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = rf_model.predict(x_test_smote_enn)

print('\n\nEvaluation with SMOTEENN and Random Forest Regression:')
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')


# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))




# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# # RandomForestRegressor modelini oluştur
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(x_train_smote_tomek, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = rf_model.predict(x_test_smote_tomek)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))


############################## Gradient Boosting Regression ####################################################################33


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# # GradientBoostingRegressor modelini oluştur
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(x_train_ros, y_train_ros)

# # Test seti üzerinde tahminler yap
# y_pred_ros = gb_model.predict(x_test_ros)

# print('\n\nEvaluation with RandomOverSampler:')
# mse = mean_squared_error(y_test_ros, y_pred_ros)
# r2 = r2_score(y_test_ros, y_pred_ros)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_ros.reshape(-1, 1))




# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# # GradientBoostingRegressor modelini oluştur
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(x_train_smote, y_train_smote)

# # Test seti üzerinde tahminler yap
# y_pred_smote= gb_model.predict(x_test_smote)

# print('\n\nEvaluation with SMOTE:')
# mse = mean_squared_error(y_test_smote, y_pred_smote)
# r2 = r2_score(y_test_smote, y_pred_smote)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote.reshape(-1, 1))



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# # GradientBoostingRegressor modelini oluştur
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(x_train_adasyn, y_train_adasyn)

# # Test seti üzerinde tahminler yap
# y_pred_adasyn = gb_model.predict(x_test_adasyn)

# print('\n\nEvaluation with ADASYN:')
# mse = mean_squared_error(y_test_adasyn, y_pred_adasyn)
# r2 = r2_score(y_test_adasyn, y_pred_adasyn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_adasyn.reshape(-1, 1))



# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# # GradientBoostingRegressor modelini oluştur
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(x_train_rus, y_train_rus)

# # Test seti üzerinde tahminler yap
# y_pred_rus = gb_model.predict(x_test_rus)

# print('\n\nEvaluation with RandomUnderSampler:')
# mse = mean_squared_error(y_test_rus, y_pred_rus)
# r2 = r2_score(y_test_rus, y_pred_rus)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_rus.reshape(-1, 1))



# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# # GradientBoostingRegressor modelini oluştur
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(x_train_nearMiss, y_train_nearMiss)

# # Test seti üzerinde tahminler yap
# y_pred_nearMiss = gb_model.predict(x_test_nearMiss)

# print('\n\nEvaluation with Near Miss Under-Sampling:')
# mse = mean_squared_error(y_test_nearMiss, y_pred_nearMiss)
# r2 = r2_score(y_test_nearMiss, y_pred_nearMiss)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_nearMiss.reshape(-1, 1))



## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# GradientBoostingRegressor modelini oluştur
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(x_train_smote_enn, y_train_smote_enn)

# Test seti üzerinde tahminler yap
y_pred_smote_enn = gb_model.predict(x_test_smote_enn)

print('\n\nEvaluation with SMOTEENN and Gradient Boosting Regression:') 
mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')


# RMSE hesapla
rmse = np.sqrt(mean_squared_error(y_test_smote_enn, y_pred_smote_enn))
print(f'RMSE: {rmse:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# # GradientBoostingRegressor modelini oluştur
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(x_train_smote_tomek, y_train_smote_tomek)

# # Test seti üzerinde tahminler yap
# y_pred_smote_tomek = gb_model.predict(x_test_smote_tomek)

# print('\n\nEvaluation with SMOTETomek:')
# mse = mean_squared_error(y_test_smote_tomek, y_pred_smote_tomek)
# r2 = r2_score(y_test_smote_tomek, y_pred_smote_tomek)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_tomek.reshape(-1, 1))
