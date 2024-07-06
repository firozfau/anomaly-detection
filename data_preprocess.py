import ast
import csv
import os
import sys
from pickle import dump, load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
import warnings

# Define output folder for processed data
output_folder = 'processed'
os.makedirs(output_folder, exist_ok=True)

def get_data_dim(dataset):
    """Return the data dimension based on the dataset name."""
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
        raise ValueError(f'Unknown dataset {dataset}')

def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
    Get data from pkl files.

    Parameters:
    - dataset: Name of the dataset.
    - max_train_size: Maximum size of training data.
    - max_test_size: Maximum size of test data.
    - print_log: Flag to print log messages.
    - do_preprocess: Flag to preprocess the data.
    - train_start: Starting index for training data.
    - test_start: Starting index for test data.

    Returns:
    - Tuple of training and test data along with test labels.
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size

    if print_log:
        print('Loading data of:', dataset)
        print("Train range:", train_start, train_end)
        print("Test range:", test_start, test_end)

    x_dim = get_data_dim(dataset)

    try:
        with open(os.path.join(output_folder, f"{dataset}_train.pkl"), "rb") as f:
            train_data = load(f).reshape((-1, x_dim))[train_start:train_end, :]
    except (FileNotFoundError, IOError) as e:
        raise FileNotFoundError(f"Training data file not found for dataset {dataset}. Error: {e}")

    try:
        with open(os.path.join(output_folder, f"{dataset}_test.pkl"), "rb") as f:
            test_data = load(f).reshape((-1, x_dim))[test_start:test_end, :]
    except (FileNotFoundError, IOError):
        test_data = None

    try:
        with open(os.path.join(output_folder, f"{dataset}_test_label.pkl"), "rb") as f:
            test_label = load(f).reshape((-1))[test_start:test_end]
    except (FileNotFoundError, IOError):
        test_label = None

    if do_preprocess:
        train_data = preprocess(train_data)
        if test_data is not None:
            test_data = preprocess(test_data)

    if print_log:
        print("Train set shape: ", train_data.shape)
        print("Test set shape: ", test_data.shape if test_data is not None else 'None')
        print("Test set label shape: ", test_label.shape if test_label is not None else 'None')

    return (train_data, None), (test_data, test_label)

def preprocess(df):
    """
    Returns normalized and standardized data.

    Parameters:
    - df: Input data frame.

    Returns:
    - Normalized data frame.
    """
    df = np.asarray(df, dtype=np.float32)
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(np.isnan(df)):
        print('Data contains null values. Replacing with 0.')
        df[np.isnan(df)] = 0

    if np.isinf(df).any():
        print('Data contains inf values. Replacing with 100.')
        df[np.isinf(df)] = 100

    # Normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def BatchSlidingWindow(values, window_length):
    """
    Create sliding windows from the input values.

    Parameters:
    - values: Input values.
    - window_length: Length of the sliding window.

    Returns:
    - Array of sliding windows.
    """
    data = [values[i:i + window_length] for i in range(len(values) - window_length)]
    return np.array(data)

def joint(values):
    """
    Concatenate values along the 0 axis.

    Parameters:
    - values: Input array.

    Returns:
    - Concatenated array.
    """
    data = np.concatenate([np.concatenate(values[i], axis=0) for i in range(values.shape[0])], axis=0)
    return data

def get_loader(values, batch_size, window_length, input_size, shuffle=False):
    """
    Get a DataLoader for the input values.

    Parameters:
    - values: Input values.
    - batch_size: Batch size.
    - window_length: Length of the sliding window.
    - input_size: Input size.
    - shuffle: Flag to shuffle the data.

    Returns:
    - DataLoader object.
    """
    if values.shape[0] % batch_size != 0:
        padding_size = batch_size - (values.shape[0] % batch_size)
        padding = np.zeros((padding_size, window_length, input_size), dtype='float32')
        values = np.concatenate((values, padding), axis=0)

    values = torch.tensor(values, dtype=torch.float32)
    return DataLoader(dataset=values, batch_size=batch_size, shuffle=shuffle)

def load_data(dataset):
    """
    Load and process data for the specified dataset.

    Parameters:
    - dataset: Name of the dataset.
    """
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
        label = np.zeros([length], dtype=bool)
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

def perform_clustering(data, subsample_size=10000, n_components=10):
    """Perform clustering on the data and return the cluster labels."""
    # Limit the number of threads used by numpy and sklearn
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Ensure the data is 2-dimensional
    if len(data.shape) != 2:
        raise ValueError("Data must be a 2-dimensional array")

    print("Data shape:", data.shape)  # Debugging print

    # Check for NaN or infinite values and handle them
    if np.any(np.isnan(data)):
        print('Data contains null values. Replacing with 0.')
        data[np.isnan(data)] = 0

    if np.isinf(data).any():
        print('Data contains inf values. Replacing with 100.')
        data[np.isinf(data)] = 100

    # Subsample the data if necessary
    if data.shape[0] > subsample_size:
        print(f"Subsampling data to {subsample_size} points.")
        data = data[np.random.choice(data.shape[0], subsample_size, replace=False), :]
        print("Subsampled data shape:", data.shape)

    # Apply PCA for dimensionality reduction
    print(f"Applying PCA to reduce to {n_components} components.")
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data)
    print("Data shape after PCA:", data_reduced.shape)

    # Determine the number of clusters based on the heuristic (square root of the number of samples)
    n_clusters = int(np.sqrt(data_reduced.shape[0]))
    if n_clusters < 2:
        n_clusters = 2

    # Limit the maximum number of clusters
    max_clusters = 50
    if n_clusters > max_clusters:
        n_clusters = max_clusters

    print(f"Number of clusters determined: {n_clusters}")  # Debugging print

    # Perform KMeans clustering
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings from sklearn
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            print("Starting KMeans fitting...")
            kmeans.fit(data_reduced)
            print("KMeans fitting completed.")
        labels = kmeans.labels_
        print("Clustering completed. Number of clusters found:", len(set(labels)))
        return labels
    except Exception as e:
        print("Error during clustering:", e)
        return None

def map_labels_to_group_names(labels, group_names):
    """Map the cluster labels to meaningful group names."""
    label_to_name = {i: group_names[i] for i in range(len(group_names))}
    names = [label_to_name[label] for label in labels]
    return names

# Example function to test perform_clustering
def test_perform_clustering():
    # Generate some sample data for testing
    np.random.seed(0)
    data = np.random.rand(100, 25)  # 100 samples, 25 features
    labels = perform_clustering(data)
    if labels is not None:
        print("Clustering labels:", labels)
    else:
        print("Clustering failed.")

if __name__ == "__main__":
    test_perform_clustering()
