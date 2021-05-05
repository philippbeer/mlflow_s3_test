# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import timeit

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def mlflow_exec(lr, test_x, test_y, alpha, l1_ratio):
    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lr, "model")


def normal_exec(lr, test_x, test_y, alpha, l1_ratio):
    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)



if __name__ == "__main__":
    mlflow.set_tracking_uri("http://139.177.179.115:5000")
    mlflow.set_experiment("test_vf_experiment")
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    runs = 1000

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

       
        stmt = 'mlflow_exec(lr, test_x, test_y, alpha, l1_ratio)'
        setup_stmt = "from __main__ import mlflow_exec, lr, test_x, test_y, alpha, l1_ratio"
        res = timeit.timeit(stmt, setup_stmt, number=runs)
        print("MLFlow run avg: {}ms".format(res/runs*1000))

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    stmt = 'normal_exec(lr, test_x, test_y, alpha, l1_ratio)'
    setup_stmt = 'from __main__ import normal_exec, lr, test_x, test_y, alpha, l1_ratio'
    res = timeit.timeit(stmt, setup_stmt, number=runs)
    print('No ML run avg: {}ms'.format(res/runs*1000))

