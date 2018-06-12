import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
from mlflow.sklearn import log_model as mlflow_sk_log_model


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "wine-quality.csv")
data = pd.read_csv(wine_path)

train, test = train_test_split(data)

train_x = train.drop(['quality'], axis=1)
test_x = test.drop(['quality'], axis=1)
train_y = train['quality']
test_y = test['quality']

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


with mlflow.start_run():
   lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
   lr.fit(train_x, train_y)

   test_pred = lr.predict(test_x)
   (rmse, mae, r2) = eval_metrics(test_y, test_pred)

   # print("pred's dimenions: {}".format(test_pred.shape))
   # print("actual's dimensions: {}".format(test_y.pred))

   print("ElasticNet model (alpha=%f. l1_ratio=%f): " %
    (alpha, l1_ratio))
   print("    RMSE: %s" % rmse)
   print("    MAE: %s" % mae)
   print("    R2: %s" % r2)

   mlflow.log_param("alpha", alpha)
   mlflow.log_param("l1_ratio", l1_ratio)
   mlflow.log_metric("rmse", rmse)
   mlflow.log_metric("r2", r2)
   mlflow.log_metric("mae", mae)

   mlflow_sk_log_model(lr, "model_lr")
