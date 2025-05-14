import os
import pickle
import random
from tqdm import tqdm

from typeformar.spacebar_detection.dataset_preparation import generate_feature_vector

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/joints")


def get_pickle_paths(dir_path):
    return [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pkl")
    ]


def load_dataset():
    dataset = []

    for path in tqdm(get_pickle_paths(DATA_DIR)):
        with open(path, "rb") as f:
            joint_dict = pickle.load(f)

        for key, value in joint_dict.items():
            feature_vectors = [
                generate_feature_vector(timestamp_joints)
                for timestamp_joints in value
                if timestamp_joints is not None and len(timestamp_joints) == 2
            ]

            dataset.append((key[0], feature_vectors))

    return dataset


def partition_dataset():
    dataset = load_dataset()
    random.shuffle(dataset)

    train_dataset = []
    test_dataset = []

    for i, pair in enumerate(dataset):
        if i % 10 == 0:
            test_dataset.append(pair)
        else:
            train_dataset.append(pair)

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = partition_dataset()
    print(len(train_dataset))
    print(len(test_dataset))
