from neuralnet import load_config, load_data, get_k_fold_ind, Neuralnetwork, train, test
import numpy as np

# Load the configuration.
config = load_config("config.yaml")
print(config)

# Load the data
x_train, y_train = load_data(path="./", mode="train")
x_test,  y_test  = load_data(path="./", mode="t10k")

indices = np.array(range(60000))
val_ind = indices[0:10000]
train_ind = indices[10000:60000]

# Create the model and train
model = Neuralnetwork(config)
plot_data = train(model, x_train[train_ind], y_train[train_ind], x_train[val_ind], y_train[val_ind], config, live_plot=False)
remodel = load_model(0)
test_acc = test(remodel, x_test, y_test)
_, test_loss = remodel(x_test, y_test)
print("Test Accuracy: {}, Test Loss: {}".format(test_acc, test_loss))