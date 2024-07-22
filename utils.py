import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import perform_clustering, map_labels_to_group_names  # Import new functions

# Define prefix for processed data folder
prefix = "processed"
normalizedStatus = False

def get_data_dim(dataset):
    """
    Get the dimensionality of the data based on the dataset name.
    """
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
        raise ValueError('Unknown dataset ' + str(dataset))

def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0, test_start=0):
    """
    Load data from pickle files and preprocess if required.

    Returns: ((train_data, None), (test_data, test_label))
    """
    # Determine the end indices for train and test sets
    train_end = None if max_train_size is None else train_start + max_train_size
    test_end = None if max_test_size is None else test_start + max_test_size

    if print_log:
        print('Loading data of:', dataset)
        print("Train range:", train_start, train_end)
        print("Test range:", test_start, test_end)

    x_dim = get_data_dim(dataset)

    # Load train data
    with open(os.path.join(prefix, dataset + '_train.pkl'), "rb") as f:
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]

    # Load test data
    try:
        with open(os.path.join(prefix, dataset + '_test.pkl'), "rb") as f:
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None

    # Load test labels
    try:
        with open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None

    # Preprocess the data if required
    if do_preprocess:
        train_data = preprocess(train_data)
        if test_data is not None:
            test_data = preprocess(test_data)

    # Log data shapes
    if print_log:
        print("Train set shape:", train_data.shape)
        if test_data is not None:
            print("Test set shape:", test_data.shape)
            if test_label is not None:
                print("Test label shape:", test_label.shape)

    return (train_data, None), (test_data, test_label)

def preprocess(df):
    global normalizedStatus

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) != 2:
        raise ValueError('Data must be a 2-D array')

    # Replace NaN and infinite values
    if np.isnan(df).any():
        print('Data contains null values. They will be replaced with 0')
        df[np.isnan(df)] = 0

    if np.isinf(df).any():
        print('Data contains inf values. They will be replaced with 100')
        df[np.isinf(df)] = 100

    # Normalize the data
    df = MinMaxScaler().fit_transform(df)
    if not normalizedStatus:
        print('Data normalized')
        print("------------------------------")

    normalizedStatus = True

    return df

def BatchSlidingWindow(values, window_length):
    """
    Generate sliding windows from the input values.
    """
    data = [values[i:i + window_length] for i in range(len(values) - window_length)]
    return np.array(data)

def joint(values):
    """
    Concatenate values along the 0 axis.
    """
    data = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            data.extend(values[i][j])
    return np.array(data)

def get_loader(values, batch_size, window_length, input_size, shuffle=False):
    """
    Create a DataLoader for the input values.
    """
    values = torch.tensor(values, dtype=torch.float32)  # Corrected the dtype

    if values.shape[0] % batch_size != 0:
        pad = torch.zeros((batch_size - values.shape[0] % batch_size, window_length, input_size), dtype=torch.float32)
        values = torch.cat((values, pad), dim=0)

    return DataLoader(dataset=values, batch_size=batch_size, shuffle=shuffle)

def load_data(f_name, f_name2):
    """
    Load true and false edges from files.
    """
    true_edge, false_edge = [], []

    with open(f_name, 'r') as f:
        for line in f:
            x, y = line.strip().split()
            true_edge.append((x, y))

    with open(f_name2, 'r') as f:
        for line in f:
            x, y = line.strip().split()
            false_edge.append((x, y))

    return true_edge, false_edge

def get_score(local_model, node1, node2):
    """
    Calculate the similarity score between two nodes.
    """
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]

        if not isinstance(vector1, np.ndarray):
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-16)
    except Exception as e:
        print(f"Error calculating score for nodes {node1} and {node2}: {e}")
        return 0

def GCN_Loss(emb):
    """
    Compute the Graph Convolutional Network (GCN) loss.
    """
    emb = emb.permute(1, 0)  # Transpose dimensions for batch processing

    true_edges, false_edges = load_data('tmp.txt', 'tmp2.txt')

    # Extract embeddings for true edges
    emb_true_first = [emb[int(edge[0])].detach().numpy() for edge in true_edges if int(edge[0]) < emb.shape[0]]
    emb_true_second = [emb[int(edge[1])].detach().numpy() for edge in true_edges if int(edge[1]) < emb.shape[0]]

    # Extract embeddings for false edges
    emb_false_first = [emb[int(edge[0])].detach().numpy() for edge in false_edges if int(edge[0]) < emb.shape[0]]
    emb_false_second = [emb[int(edge[1])].detach().numpy() for edge in false_edges if int(edge[1]) < emb.shape[0]]

    # Calculate dot products
    T1 = np.dot(np.array(emb_true_first), np.array(emb_true_second).T)
    T2 = np.dot(np.array(emb_false_first), np.array(emb_false_second).T)

    # Convert results to tensors
    pos_out = torch.tensor(np.diag(T1), dtype=torch.float32)
    neg_out = torch.tensor(np.diag(T2), dtype=torch.float32)

    # Ensure pos_out and neg_out have the same dimensions
    min_len = min(pos_out.shape[0], neg_out.shape[0])
    pos_out = pos_out[:min_len]
    neg_out = neg_out[:min_len]

    # Calculate loss
    loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

    return loss
