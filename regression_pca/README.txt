This file will briefly give an overview of each .py file and its purpose

1. classifier.py - This file contains two classes namely LogisticRegression and SoftmaxRegression. Both classes have the predict, get_loss, and get_update methods which are quite self explanatory

2. dataloader.py - This file contains the Dataloader class which loads the data, divides it into k-folds, computes PCA componnets of training set, projects the test and val set on PCA components of training set.

3. trainer.py - This is the file which first loads the data and model. Next, it uses the data to train the model by either BGD or SGD and then return all the error and accuracy data. It also computes the confusion matrix.

4. main.py - Run this file to run an example for either Logistic or Softmax Regression
