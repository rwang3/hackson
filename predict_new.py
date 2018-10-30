fimport os
from collections import Counter
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import sklearn

test_csv = "orange_small_train_appetency.csv"


columns = joblib.load("preprocess/columns.pkl")
self_response_column = columns[len(columns) - 1]
self_select_columns = columns[0:len(columns) - 1]
self_predict_data = pd.read_csv(test_csv)
self_model = joblib.load("ml/ml.pkl")
self_feature_types = ["numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "numerical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "numerical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "numerical"]

data = pd.DataFrame(self_predict_data, columns=self_select_columns)
ar = data.values

for index in range(len(self_select_columns)):
    if self_feature_types[index] == 'categorical':
        isnan_ar = np.array([x is np.nan for x in ar[:, index]])
        contains_nan = isnan_ar.any()
        if contains_nan:
            ar[isnan_ar, index] = 'nan'
        model = joblib.load("preprocess/"+self_select_columns[index]+".pkl")
        ar[:, index] = model.transform(ar[:, index])
        if contains_nan:
            ar[isnan_ar, index] = np.nan
X = ar
prediction = self_model.predict(X)
self_predict_data[self_response_column + "_prediction"] = prediction

metrics = {
    'accuracy': sklearn.metrics.accuracy_score(response, predict),
    'f1': sklearn.metrics.f1_score(response, predict, average="macro"),
    'recall': sklearn.metrics.recall_score(response, predict, average="macro"),
    'precision': sklearn.metrics.precision_score(response, predict, average="macro"),
    'ber': autosklearn.metrics.classification_metrics.balanced_classification_error_rate(response, predict)
}
print(metrics)
# 保存结果：
self_predict_data.to_csv('prediction.csv', index=False)