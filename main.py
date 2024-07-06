import torch
import numpy as np
import torch.optim as optim
from utils import get_data, get_data_dim, get_loader, perform_clustering, map_labels_to_group_names
from eval_method import bf_search
from tqdm import tqdm
from Model import MUTANT


class ExpConfig():
    dataset = "SMAP"
    val = 0.35
    max_train_size = None
    train_start = 0
    max_test_size = None
    test_start = 0
    input_dim = get_data_dim(dataset)
    batch_size = 120
    out_dim = 5
    window_length = 20
    hidden_size = 100
    latent_size = 100
    N = 256


def main():
    config = ExpConfig()

    print("Loading data...")
    (train_data, _), (test_data, test_label) = get_data(
        config.dataset, config.max_train_size, config.max_test_size,
        train_start=config.train_start, test_start=config.test_start
    )

    print("Splitting data into train, validation, and test sets...")
    n = int(test_data.shape[0] * config.val)
    val_data = test_data[-n:]
    val_label = test_label[-n:]
    test_data = test_data[:-n]
    test_label = test_label[:-n]

    print("Creating sliding windows...")
    train_data = train_data[
        np.arange(config.window_length)[None, :] + np.arange(train_data.shape[0] - config.window_length)[:, None]]
    val_data = val_data[
        np.arange(config.window_length)[None, :] + np.arange(val_data.shape[0] - config.window_length)[:, None]]
    test_data = test_data[
        np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]

    num_val = int(val_data.shape[0] / config.batch_size)
    con_val = val_data.shape[0] % config.batch_size
    num_t = int(test_data.shape[0] / config.batch_size)
    con_t = test_data.shape[0] % config.batch_size

    w_size = config.input_dim * config.out_dim

    print("Creating data loaders...")
    train_loader = get_loader(train_data, batch_size=config.batch_size, window_length=config.window_length,
                              input_size=config.input_dim, shuffle=True)
    val_loader = get_loader(val_data, batch_size=config.batch_size, window_length=config.window_length,
                            input_size=config.input_dim, shuffle=True)
    test_loader = get_loader(test_data, batch_size=config.batch_size, window_length=config.window_length,
                             input_size=config.input_dim, shuffle=False)

    print("Initializing model...")
    model = MUTANT(config.input_dim, w_size, config.hidden_size, config.latent_size, config.batch_size,
                   config.window_length, config.out_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    save_path = 'model.pt'
    flag = 0
    f1 = -1
    numberProcess = 10
    dataSynchronizationStatus = False

    if dataSynchronizationStatus:
        message = f"\n Please wait .... (data process will be {numberProcess} times) \n\n"
        print(message)

        for epoch in range(numberProcess):
            l = 0
            i = 0
            subMessage = f"Data scanning-{epoch + 1}:"
            print(subMessage)

            for inputs in tqdm(train_loader):
                loss = model(inputs)
                loss.backward()
                if (i % config.N == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                i += 1

            if flag == 1:
                model.load_state_dict(torch.load(save_path))
            val_score = model.is_anomaly(val_loader, num_val, con_val)
            t, th = bf_search(val_score, val_label[-len(val_score):], step_num=700)

            if (t[0] > f1):
                f1 = t[0]
                torch.save(model.state_dict(), save_path)
                flag = 1

        model.load_state_dict(torch.load(save_path))

    print("Evaluating model on test data...")
    test_score = model.is_anomaly(test_loader, num_t, con_t)

    print("Performing brute-force search for the best threshold...")
    t, th = bf_search(test_score, test_label[-len(test_score):], step_num=700)

    print("Performing clustering on test data...")
    try:
        test_data_reshaped = test_data.reshape(-1, config.input_dim)
        print("Data reshaped for clustering:", test_data_reshaped.shape)  # Debugging print
        test_cluster_labels = perform_clustering(test_data_reshaped)
        unique_labels = np.unique(test_cluster_labels)
        group_names = [f"Group{i}" for i in unique_labels]
        test_group_names = map_labels_to_group_names(test_cluster_labels, group_names)

        print('********************* Final result of Test data ********************************\n')
        print('Dataset-Name:', config.dataset)
        print('Threshold-value:', th)
        print("True Positive:", t[3])
        print("False Positive:", t[5])
        print("False Negative:", t[6])
        print("Precision:", t[1])
        print("Recall:", t[2])
        print("F1-Score:", t[0])

        # Display cluster information
        unique, counts = np.unique(test_group_names, return_counts=True)
        print("Number of groups:", len(unique))
        print("Group names and counts:", dict(zip(unique, counts)))
    except Exception as e:
        print("Error during clustering process:", e)

    return t[1], t[2], t[0]


if __name__ == '__main__':
    main()
