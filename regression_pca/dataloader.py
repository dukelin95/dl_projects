class dataloader():

    def __init__(self, filelocation):
        self.filelocation = filelocation

    # DONE ONLY ON TRAINING SET
    def pca(self, num_comp, training_data):

        self.e_vec = ...
        self.e_val = ...

        self.processed_data = ...

        return self.processed_data, self.e_val, self.e_vec

    def get_k_fold(self,k ):

       return [trainings, validations, tests]

    def get_80_10_10(self):
        # use get_k_fold
        return training, validation, test
