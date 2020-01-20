from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
from itertools import combinations_with_replacement as cwr
from numpy import linalg as LA
import matplotlib.pyplot as plt


class Dataloader:

    def __init__(self, filelocation):
        self.filelocation = filelocation
        self.load_data()

    def load_data(self):
        """
        Load all PNG images stored in your data directory into a list of NumPy
        arrays.

        Returns:
            images: A dictionary with keys as emotions and a list containing images associated with each key.
            cnt: A dictionary that stores the # of images in each emotion
        """
        images = defaultdict(list)

        # Get the list of emotional directory:
        for e in listdir(self.filelocation):
            # excluding any non-directory files
            if not os.path.isdir(os.path.join(self.filelocation, e)):
                continue
            # Get the list of image file names
            all_files = listdir(os.path.join(self.filelocation, e))

            for file in all_files:
                # Load only image files as PIL images and convert to NumPy arrays
                if '.png' in file:
                    img = Image.open(os.path.join(self.filelocation, e, file))
                    images[e].append(np.array(img))

        print("Emotions: {} \n".format(list(images.keys())))

        cnt = defaultdict(int)
        for e in images.keys():
            print("{}: {} # of images".format(e, len(images[e])))
            cnt[e] = len(images[e])

        self.images = images
        self.cnt = cnt
        self.image_size = images['fear'][0].shape[0] * images['fear'][0].shape[1]

    def balanced_sampler(self, emotions):
        # this ensures everyone has the same balanced subset for model training, don't change this seed value
        random.seed(20)
        print("\nBalanced Set:")
        min_cnt = min([self.cnt[e] for e in emotions])
        balanced_subset = defaultdict(list)
        for e in emotions:
            balanced_subset[e] = copy.deepcopy(self.images[e])
            random.shuffle(balanced_subset[e])
            balanced_subset[e] = balanced_subset[e][:min_cnt]
            print('{}: {} # of images'.format(e, len(balanced_subset[e])))
        total_imgs = min_cnt * len(emotions)

        return balanced_subset, total_imgs, min_cnt

    def display_face(self, img):
        """
        Display the input image and optionally save as a PNG.

        Args:
            img: The NumPy array or image to display

        Returns: None
        """
        # Convert img to PIL Image object (if it's an ndarray)
        if type(img) == np.ndarray:
            print("Converting from array to PIL Image")
            img = Image.fromarray(img)

        # Display the image
        img.show()

    def pca(self, data, p):
        '''
        WARNING: SHOULD DONE ONLY ON TRAINING SET
        
        Implements PCA (using Turk and Pentland trick) and returns the top p eigen values and vectors
        Assumes data is of shape (M x (hxw)), where M = number of images of height h and width w

        Returns:
        data_reduced = size (Mxp), where M is number of images
        top_p_eig_values = size (p)
        top_p_eig_vectors = size (dxp), each column is a d dimensional eigen vector
        '''
        assert isinstance(data, np.ndarray)
        assert isinstance(p, int) and p > 0
        assert np.max(data) <= 1.0, 'pixel value range should be 0 to 1'

        if data.shape[0] > data.shape[1]:
            print('CAUTION: number of images > number of dimensions. Might want to transpose data matrix!')

        M = data.shape[0] # M = number of images
        A = data.reshape(M, -1) # A = (Mxd)
        
        # Subtracing mean
        mean = np.mean(A, axis=0)
        A = A - mean #subtracing mean face from all data
        A = A.T # changing shape from (Mxd) to (dxM)

        # Eigen analysis
        eig_values, eig_vectors = LA.eig(A.T@A) #each column of eig_vectors is an eigen vector
        eig_vectors = A @ eig_vectors # TURK AND PENTLAND trick (dxM) x (MxM) = (dxM)
        
        sort_index = list(np.argsort(eig_values)) #sorting eigen values
        sort_index = sort_index[::-1] #descending order
        eig_values_sorted = eig_values[sort_index]
        eig_vectors_sorted = eig_vectors[:, sort_index] #sorting eigen vectors according to eigen values
        
        # Picking top p eigen values and vectors
        top_p_eig_values = eig_values_sorted[0:p] #size [p,]
        top_p_eig_vectors = eig_vectors[:, 0:p] #size [43008, 15]

        # Projecting data on to top p eigen vectors
        data_reduced = (A.T + mean) @ top_p_eig_vectors

        self.top_p_eig_vectors = top_p_eig_vectors
        self.train_mean = mean

        return data_reduced, top_p_eig_values, top_p_eig_vectors

    def project_pca(self, A):
        return (A - self.train_mean) @ self.top_p_eig_vectors

    def process_batch(self, data):
        """
        Vectorize and normalize batch and add 1's for affine model
        :param data: 3-D data
        :return: 2-D data
        """
        batch_size = data.shape[0]
        vectorized = data.reshape(batch_size, -1)/255
        return vectorized

    def get_k_fold(self, k, emotions):
        """
        Get the data set split up for cross validation
        :param k: number of folds
        :param emotions: list of strings of emotions
        :return: tuple of lists (each corresponding to set) that contain a tuple of (batch of pictures, batch of target)
        """

        # dictionary to change emotion to a target value
        enc_emo = {emotion: num for num, emotion in enumerate(emotions)}

        # use an edited balanced_sampler to get data and split each emotion evenly (roughly)
        images, total_num_imgs, len_emotions = self.balanced_sampler(emotions)
        emotions_split = {key: np.array_split(value, k) for key, value in images.items()}

        # combine a split from each emotion together, and the target values as well
        order = [0 if a % 2 == 0 else -1 for a in range(len(emotions_split.keys()))]
        splits = []
        targets = []
        for fold in range(k):
            image_set = []
            target_set = []
            for i, key in enumerate(emotions_split.keys()):
                selection = emotions_split[key].pop(order[i])
                image_set.append(selection)
                target_set.append(np.zeros((len(selection), 1)) + enc_emo[key])
            splits.append(np.concatenate(image_set))
            targets.append(np.concatenate(target_set))

        # organize the splits into k ways, where each split is a test and val set at least once
        # also has targets, so list of tuples (image, target)
        trainings = []
        validations = []
        tests = []
        for test_ind, val_ind in [(a, a-1) for a in list(range(k))]:
            indexes = list(range(k))
            validations.append((self.process_batch(splits[val_ind]), targets[val_ind]))
            tests.append((self.process_batch(splits[test_ind]), targets[test_ind]))
            indexes.remove(val_ind if val_ind != -1 else k - 1)
            indexes.remove(test_ind)
            trainings.append((self.process_batch(np.concatenate([splits[i] for i in indexes])), np.concatenate([targets[i] for i in indexes])))

        return trainings, validations, tests

    def get_80_10_10(self, emotions):
        """ Use get_k_fold to do a 80/10/10 percent split

        :param emotions: list of emotion strings
        :return: tuple of data split
        """
        tr, val, te = self.get_k_fold(10, emotions)
        return tr[0], val[0], te[0]


if __name__ == '__main__':
    dl = Dataloader("./facial_expressions_data/aligned/")
    tr, va, te = dl.get_k_fold(10, ['happiness', 'anger', 'disgust'])
     ## ---- Testing PCA ------
    data1, target1 = tr[0]
    p = 15
    data1_reduced, eig_values, eig_vectors = dl.pca(data1, p)

    # Plotting eigenfaces
    # fig, axs = plt.subplots(3, 5)
    # axs = axs.flatten()
    # for i in range(p):
    #     eig_face = eig_vectors[:,i].reshape(224,-1)
    #     axs[i].imshow(eig_face, cmap='gray')
    # plt.show()
    ## ---- PCA testing ended ------