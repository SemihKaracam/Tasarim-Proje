

## SMOTEEN dengeleme yöntemiyle dengelenen veri seti ile Rasgele Orman Regresyonu modeli eğitildi.


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder  #preprocessing
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split    #splitting data

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTEENN


#Duzenlenmis veri seti icin
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
 

## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# Random Forest Regressor modeli için parametre aralığını belirleme
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

# K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# RMSE hesaplama için özel bir skor fonksiyonu oluşturma
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)


# Random Forest Regressor modeli
forest_model = RandomForestRegressor()

# GridSearchCV kullanarak hiperparametre optimizasyonu
grid_search = GridSearchCV(forest_model, param_grid, cv=kf, scoring=rmse_scorer)
grid_search.fit(x_train_smote_enn, y_train_smote_enn.ravel())

# En iyi hiperparametreleri ve modeli alın
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Cross-validation ile RMSE hesaplama
cv_rmse = np.sqrt(-grid_search.best_score_)


print('\n\nEvaluation with SMOTEENN and Random Forest Regression:')

print(f'En iyi hiperparametreler: {best_params}') #En iyi hiperparametreler: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
y_pred_smote_enn = best_model.predict(x_test_smote_enn)

mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))


# İlk birkaç örnek çıktıyı kontrol et
print("Eğitim örnekleri için gerçek değerler:")
print(y_test_smote_enn[10:20])

print("Tahmin edilen değerlerin yüzde olarak karşılığı:")
# print(y_pred[:5])
print(y_pred_scaled[10:20])


lastMyList:  ['1', '22', '0', '0', '0', '0', '0', '21', '0', 80]
print("lastMyList: ",lastMyList)
mlist = np.array(lastMyList)
tahmin2 = best_model.predict(mlist.reshape(1,-1))
