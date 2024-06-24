import ast
import csv
import os
import sys
from pickle import dump
import pandas as pd
import numpy as np

output_folder = 'processed'
os.makedirs(output_folder, exist_ok=True)

def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif dataset == 'SWaT':
        return 51
    elif dataset == 'WADI':
        return 123
    else:
        raise ValueError('unknown dataset '+str(dataset))

def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0, test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(output_folder, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(output_folder, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(output_folder, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)

    return (train_data, None), (test_data, test_label)

def preprocess(df):
    """returns normalized and standardized data."""
    df = np.asarray(df, dtype=np.float32)
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        inx = np.isnan(df)
        df[inx] = 0
    if np.isinf(df).any():
        inx = np.isinf(df)
        print('Data contains inf values. Will be replaced with 100')
        df[inx] = 100
    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def BatchSlidingWindow(values, window_length):
    data = []
    for i in range(len(values) - window_length):
        data.append(values[i:i + window_length])
    data = np.array(data)
    return data

def joint(values):
    data = []
    for i in range(values.shape[0]):
        a = []
        for j in range(values.shape[1]):
            a = np.concatenate((a, values[i][j]), axis=0)
        data = np.concatenate((data, a), axis=0)
    return data

def get_loader(values, batch_size, window_length, input_size, shuffle=False):
    if values.shape[0] % batch_size !=0:
        for i in range(batch_size-values.shape[0] % batch_size):
            a = torch.tensor(np.zeros((1, window_length, input_size), dtype='float32'))
            values = np.concatenate((values, a), axis=0)
    values = torch.tensor(values)
    return DataLoader(dataset=values, batch_size=batch_size, shuffle=shuffle)

def load_data(dataset):
    dataset_folder = 'data'
    labeled_anomalies_file = os.path.join(dataset_folder, 'labeled_anomalies.csv')

    if not os.path.exists(labeled_anomalies_file):
        print(f"Error: File '{labeled_anomalies_file}' not found.")
        sys.exit(1)

    with open(labeled_anomalies_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])
    label_folder = os.path.join(dataset_folder, 'test_label')
    os.makedirs(label_folder, exist_ok=True)
    data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
    labels = []
    for row in data_info:
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=bool)  # Change dtype to bool or np.bool_
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = True
        labels.extend(label)
    labels = np.asarray(labels)
    print(dataset, 'test_label', labels.shape)
    with open(os.path.join(output_folder, f"{dataset}_test_label.pkl"), "wb") as file:
        dump(labels, file)

    def concatenate_and_save(category):
        data = []
        for row in data_info:
            filename = row[0]
            temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
            data.extend(temp)
        data = np.asarray(data)
        print(dataset, category, data.shape)
        with open(os.path.join(output_folder, f"{dataset}_{category}.pkl"), "wb") as file:
            dump(data, file)

    for c in ['train', 'test']:
        concatenate_and_save(c)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Please provide a dataset name (SMAP, MSL, SWaT, WADI) as argument.")
        sys.exit(1)

    dataset = sys.argv[1]
    if dataset not in ['SMAP', 'MSL', 'SWaT', 'WADI']:
        print("Error: Invalid dataset name. Choose from SMAP, MSL, SWaT, WADI.")
        sys.exit(1)

    load_data(dataset)
