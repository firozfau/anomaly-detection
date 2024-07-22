#Graph.py
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np
from utils import BatchSlidingWindow, get_data, get_data_dim


def construct_graph(features, topk):
    """
    Construct a graph based on the correlation of features.
    Creates two files: one for positive edges (similar nodes) and one for negative edges (dissimilar nodes).

    Args:
        features (np.ndarray): Feature matrix.
        topk (int): Number of top similar and dissimilar nodes to consider.
    """
    fname = 'tmp.txt'
    fname2 = 'tmp2.txt'

    # Compute the correlation matrix
    dist = np.corrcoef(features)

    # Initialize lists to store indices of top similar and dissimilar nodes
    inds = []
    negs = []

    for i in range(dist.shape[0]):
        # Get indices of topk similar and dissimilar nodes
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        neg = np.argpartition(dist[i, :], (topk + 1))[:topk + 1]
        inds.append(ind)
        negs.append(neg)

    # Write positive edges to file
    with open(fname, 'w') as f:
        for i, v in enumerate(inds):
            for vv in v:
                if vv != i:
                    f.write('{} {}\n'.format(i, vv))

    # Write negative edges to file
    with open(fname2, 'w') as f2:
        for i, v in enumerate(negs):
            for vv in v:
                if vv != i:
                    f2.write('{} {}\n'.format(i, vv))


def normalize(mx):
    """
    Row-normalize sparse matrix.

    Args:
        mx (sp.coo_matrix): Sparse matrix to normalize.

    Returns:
        sp.coo_matrix: Row-normalized sparse matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # Replace infinities with zeros
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def generate_knn(data):
    """
    Generate K-Nearest Neighbors graph for the given data.

    Args:
        data (np.ndarray): Data to generate KNN graph for.
    """
    topk = 6  # Number of nearest neighbors
    construct_graph(data, topk)


def returnA(x):
    """
    Generate the adjacency matrix for the feature graph.

    Args:
        x (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Normalized adjacency matrix.
    """
    x = np.array(x).T  # Transpose the input data
    generate_knn(x)  # Generate KNN graph

    # Load feature edges from file
    featuregraph_path = 'tmp.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)

    # Create sparse adjacency matrix
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(x.shape[0], x.shape[0]),
                         dtype=np.float32)
    fadj = fadj + sp.coo_matrix(np.eye(x.shape[0]))  # Add self-loops
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)  # Symmetrize the matrix
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))  # Normalize the adjacency matrix
    nfadj = nfadj.A  # Convert to dense format

    return nfadj
