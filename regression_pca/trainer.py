
class trainer():

    def __init__(self, classifier, dataloader):
        self.classifier = classifier
        self.dataloader = dataloader
        self.weights = weight_init()
        pass

    def weight_init(self):

    def plot(self):
        # average training and holdout error and standard deviation each epoch
        pass

    def save_model(self):

        pass

    def evaluate(self, data):
        return val_error

    def train(self, lr, num_epochs, k=10):
        test_errors = []
        [trainings, validations, tests] = self.dataloader.get_k_fold(k)
        for fold in k:
            train = trainings[fold]
            val = validations[fold]
            test = tests[fold]
            val_errors = []
            for epoch in num_epochs:
                update = self.classifier.get_step(...)
                self.weights = self.weights + update
                val_errors.append(self.evaluate(val))

                if check_for_lowval_error:
                    self.save_model()

            # use best weights to calculate test error on test set
            self.evaluate(test)
        pass

    def do_other_plots(self):
        self.classifier.plot()

# classifiers below
class logistic_regression():
    pass
    # add whatever function you need to plot

    def get_step(self):
        return gradient_step

class softmax_regression():

    def __init__(self, method):
        self.gradient_method = method
    pass
    # add whatever function you need to plot

    def get_step(self):
        # do stochastic or batch




