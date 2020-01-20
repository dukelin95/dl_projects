import numpy as np
import matplotlib.pyplot as plt
import pickle

class trainer():

    def __init__(self, classifier, dataloader, emotions, method='batch'):
        self.classifier = classifier
        self.dataloader = dataloader
        self.emotions = emotions
        self.method = method

        self.target_emote = {num: emotion for num, emotion in enumerate(emotions)}
        self.emote_target = {emotion: num for num, emotion in enumerate(emotions)}

    def weight_init(self, weight_len):
        """
        Initialize weight to 0
        :param weight_len: size required
        :return: array of zeros
        """
        return np.zeros((weight_len + 1, 1))

    def fold_plot(self):
        # TODO Updating plot?
        # average training and holdout error and standard deviation each fold
        pass

    def live_plotter(self, x_vec, y1_data, line1, y2_data, line2, identifier='', pause_time=0.01):
        if line1 == []:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
            line2, = ax.plot(x_vec, y2_data, '-o', alpha=0.8)
            line1.set_label("Test")
            line2.set_label("Validation")
            # update plot label/title
            plt.ylabel('Loss')
            plt.title('{}'.format(identifier))
            ax.legend()
            plt.show()

        # after the figure, axis, and line are created, we only need to update the y-data
        line1.set_ydata(y1_data)
        line2.set_ydata(y2_data)
        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
        if np.min(y2_data) <= line1.axes.get_ylim()[0] or np.max(y2_data) >= line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y2_data) - np.std(y2_data), np.max(y2_data) + np.std(y2_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1, line2

    def save_model(self, fold, weights):
        """
        Saves the model as a .npy file

        :param fold: fold currently on
        :param weights: weights to be saved
        :return: nothing
        """
        np.save('{}_weights_{}.npy'.format(self.classifier.__class__.__name__, fold, self.method), weights)

    def load_model(self):
        pass

    def evaluate(self, weights, data, targets):
        prediction, actual_val = self.classifier.predict(weights, data)
        loss = self.classifier.get_loss(actual_val, targets)

        accuracy = np.sum((prediction == targets).astype(int))/data.shape[0]
        return loss, accuracy

    def train(self, lr, num_epochs, num_pca_comps=10, k=10):
        """
        # TODO MUST CHANGE TARGETS ACCORDINGLY 111!!!
        """

        # load data, init weights
        trainings, validations, tests = self.dataloader.get_k_fold(k, self.emotions)
        test_eval = []
        val_eval = []
        best_losses = []

        # for cross validation
        for fold in range(k):
            weights = self.weight_init(num_pca_comps)
            tr_data, tr_targets = trainings[fold]
            val_data, val_targets = validations[fold]
            te_data, te_targets = tests[fold]
            fold_test_eval = []
            fold_val_eval = []
            val_loss_threshold = np.inf

            # pca for training data, bias for all data
            tr_data, _, _ = self.dataloader.pca(tr_data, num_pca_comps)
            tr_data = np.concatenate((tr_data, np.ones((tr_data.shape[0], 1))), axis=1)
            te_data = np.concatenate((self.dataloader.project_pca(te_data), np.ones((te_data.shape[0], 1))), axis=1)
            val_data = np.concatenate((self.dataloader.project_pca(val_data), np.ones((val_data.shape[0], 1))), axis=1)

            # graphing utility
            x_vec = np.linspace(1, num_epochs, num_epochs)
            val_vec = np.zeros(len(x_vec))
            test_vec = np.zeros(len(x_vec))
            test_line = []
            val_line = []
            for epoch in range(num_epochs):
                if self.method == 'sgd':
                    raise NotImplementedError("Implement SGD")
                elif self.method == 'batch':
                    # update
                    prediction, actual_val = self.classifier.predict(weights, tr_data)
                    print("Prediction values: {}".format(actual_val.T))
                    update = self.classifier.get_update(actual_val, tr_data, tr_targets)
                    print("Update: {}".format(update.T))
                    weights = weights - (lr * update)  # Gradient DESCENT not ascent
                    # print("Weights: {}".format(weights.T))

                    # get respective (loss, acc)
                    test_loss, test_acc = self.evaluate(weights, te_data, te_targets)
                    fold_test_eval.append((test_loss, test_acc))
                    val_loss, val_acc = self.evaluate(weights, val_data, val_targets)
                    fold_val_eval.append((val_loss, val_acc))
                    # print("Val loss: {}, val acc: {}".format(val_loss, val_acc))

                    test_vec[epoch] = test_loss
                    val_vec[epoch] = val_loss
                    test_line, val_line = self.live_plotter(x_vec, test_vec, test_line, val_vec, val_line, "{} Loss".format(fold))

                    # save best model based on loss
                    if val_acc < val_loss_threshold:
                        print("Fold {}: saved model on epoch: {}".format(fold, epoch))
                        val_loss_threshold = fold_val_eval[-1][0]
                        self.save_model(fold, weights)
                        best_weights = weights

                else:
                    raise ValueError("sgd and batch are only methods supported")

            # get best loss and best accuracy
            # TODO idk whats wrong
            # prediction = self.classifier.predict(, tr_data) #???? why predict on tr_data
            # best_loss, best_acc = self.evaluate(prediction, te_targets)
            # print("For fold #{}, the best loss and accuracy on test set".format(fold+1, best_loss, best_acc))
            # self.fold_plot()

            test_eval.append(fold_test_eval)
            val_eval.append(fold_val_eval)
            # best_losses.append(best_loss)


if __name__ == '__main__':
    from dataloader import Dataloader
    from classifiers import LogisticRegression

    lr = 1e-6
    num_epochs = 25
    num_pca_comps = 30
    k = 5

    dl = Dataloader("./facial_expressions_data/aligned/")
    emotions = ['anger', 'happiness']
    method = 'batch'
    cl = LogisticRegression()

    trainer = trainer(cl, dl, emotions, method)
    trainer.train(lr, num_epochs, num_pca_comps, k)
