from dataloader import Dataloader
from classifiers import LogisticRegression
from classifiers import SoftmaxRegression
from trainer import Trainer

# main file to allow testing of code
# much more detail in the other class files

lr = 1e-4
num_epochs = 50
num_pca_comps = 30
k = 10

# set up for softmax regression
dl = Dataloader("./facial_expressions_data/aligned/")
emotions = ['anger', 'happiness', 'disgust', 'sadness']
method = 'sgd'
cl = SoftmaxRegression(len(emotions))

trainer = Trainer(cl, dl, emotions, method)
trainer.train(lr, num_epochs, num_pca_comps, k)

# set up for logistic regression
dl = Dataloader("./facial_expressions_data/aligned/")
emotions = ['anger', 'happiness']
method = 'batch'
cl = LogisticRegression()

trainer = Trainer(cl, dl, emotions, method)
trainer.train(lr, num_epochs, num_pca_comps, k)
