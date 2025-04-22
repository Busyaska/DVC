import yaml
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_model(config):
    train_dataset_path = config["train_model"]["train_dataset_path"]
    test_dataset_path = config["train_model"]["test_dataset_path"]
    best_model_path = config["train_model"]["best_model_path"]

    train_df = pd.read_csv(train_dataset_path)
    test_df = pd.read_csv(test_dataset_path)
    X_train = train_df.drop('Anxiety_Level_(1-10)', axis=1)
    Y_train = train_df['Anxiety_Level_(1-10)']
    X_test = test_df.drop('Anxiety_Level_(1-10)', axis=1)
    Y_test = test_df['Anxiety_Level_(1-10)']
    

    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
            'l1_ratio': [0.001, 0.05, 0.01, 0.2],
            "penalty": ["l1","l2","elasticnet"],
            "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
            "fit_intercept": [False, True],
            }
    
    mlflow.set_experiment("Anxiety Level Experiment")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv = 3, n_jobs = 4)
        clf.fit(X_train, Y_train)
        best = clf.best_estimator_
        y_pred = best.predict(X_test)
        (rmse, mae, r2)  = eval_metrics(Y_test, y_pred)
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        penalty = best.penalty
        eta0 = best.eta0
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("eta0", eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model'
    
    with open(best_model_path, 'w') as f:
        f.write(path2model)


if __name__ == "__main__":
    with open('src/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    train_model(config)
