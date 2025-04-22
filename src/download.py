import kagglehub
import shutil
import os
import yaml


def download_dataset(config):
    destination = config["download"]["dataset_dir"]
    os.makedirs(destination, exist_ok=True)
    path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset", force_download=True)

    for file_name in os.listdir(path):
        shutil.move(os.path.join(path, file_name), os.path.join(destination, file_name))


if __name__ == '__main__':
    with open('src/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    download_dataset(config)
