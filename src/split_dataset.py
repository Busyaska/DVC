import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(config):
    dataset_path = config["split_dataset"]["dataset_path"]
    train_dataset_path = config["split_dataset"]["train_dataset_path"]
    test_dataset_path = config["split_dataset"]["test_dataset_path"]
    dataset = pd.read_csv(dataset_path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, train_size=0.75, random_state=52)

    train_dataset.to_csv(train_dataset_path, index=False)
    test_dataset.to_csv(test_dataset_path, index=False)


if __name__ == '__main__':
    with open('src/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    split_dataset(config)