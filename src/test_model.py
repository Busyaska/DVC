import yaml
import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from dvclive import Live

def test_model(config):
    test_dataset_path = config["test_model"]["test_dataset_path"]
    best_model_path = config["test_model"]["best_model_path"]

    df = pd.read_csv(test_dataset_path)
    X = df.drop('Anxiety_Level_(1-10)', axis=1)
    Y = df["Anxiety_Level_(1-10)"]

    with open(best_model_path, 'r') as path_file:
        path = path_file.readline().strip()[1:] + '/model.pkl'
        model = joblib.load(path)
    
    prediction = model.predict(X)

    r2 = r2_score(Y, prediction)
    mae = mean_absolute_error(Y, prediction)
    rmse = root_mean_squared_error(Y, prediction)
    
    with Live() as live:
        live.log_metric("r2", r2)
        live.log_metric("mae", mae)
        live.log_metric("rmse", rmse)


if __name__ == "__main__":
    with open('src/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    test_model(config)
