from neuralnet import load_config, load_data, get_k_fold_ind, Neuralnetwork, train, test
import numpy as np

# Load the configuration.
config = load_config("config.yaml")

# Load the data
x_train, y_train = load_data(path="./", mode="train")
x_test,  y_test  = load_data(path="./", mode="t10k")

cross_val_indices = get_k_fold_ind(10, x_train)
for i, v in enumerate(cross_val_indices):
    train_ind = cross_val_indices.copy()
    val_ind = train_ind.pop(i)
    train_ind = np.concatenate(train_ind)

    # Create the model and train
    model = Neuralnetwork(config)
    train(model, x_train[train_ind], y_train[train_ind], x_train[val_ind], y_train[val_ind], config, i, live_plot=True)
    test_acc = test(model, x_test, y_test)
    print("Fold {}'s Test Accuracy: {}".format(i ,test_acc))
