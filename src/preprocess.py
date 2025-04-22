import pandas as pd
import yaml


def preprocess_dataset(config):
    origin_dataset_path = config["preprocess"]["origin_dataset_path"]
    result_dataset_path = config["preprocess"]["result_dataset_path"]

    df = pd.read_csv(origin_dataset_path)

    df.rename(lambda x: x if ' ' not in x else x.replace(' ', '_'), axis='columns', inplace=True)

    numeric_columns = df.select_dtypes('number').columns

    for column in numeric_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)

    df['Smoking'] = pd.get_dummies(df.Smoking).astype('Int64')['Yes']
    df['Family_History_of_Anxiety'] = pd.get_dummies(df.Family_History_of_Anxiety).astype('Int64')['Yes']
    df['Dizziness'] = pd.get_dummies(df.Dizziness).astype('Int64')['Yes']
    df['Medication'] = pd.get_dummies(df.Medication).astype('Int64')['Yes']
    df['Recent_Major_Life_Event'] = pd.get_dummies(df.Recent_Major_Life_Event).astype('Int64')['Yes']

    one_hot_occupation = pd.get_dummies(df['Occupation'], dtype='Int64')
    df = pd.concat([df, one_hot_occupation], axis=1)
    df.drop('Occupation', axis=1, inplace=True)

    one_hot_gender = pd.get_dummies(df['Gender'], dtype='Int64')
    df = pd.concat([df, one_hot_gender], axis=1)
    df.drop('Gender', axis=1, inplace=True)

    df.to_csv(result_dataset_path, index=False)


if __name__ == '__main__':
    with open('src/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    preprocess_dataset(config)