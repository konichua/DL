import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.names = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer",
                      5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
        self.labels = self._load_labels()
        self.epoch = 0
        self.rem_batch_size = 1
        self.batch_count = batch_size
        self.s_ind = 0
        self.f_ind = batch_size
        self.index = self._make_index()

    def _load_labels(self):
        with open(self.label_path) as f:
            load = json.load(f)
            labels = {}
            for key, value in load.items():
                labels[int(key)] = int(value)
            return labels

    def _make_index(self):
        index = np.array(np.arange(100))
        if self.shuffle:
            np.random.shuffle(index)
        return index

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        images = []
        labels = []

        while self.batch_count:
            for npy in self.index[self.s_ind:]:
                self.batch_count -= 1
                image = resize(np.load(f'{self.file_path}{npy}.npy'), tuple(self.image_size))
                images.append(self.augment(image))
                labels.append(self.labels.get(npy))
                if self.batch_count == 0:
                    self.s_ind = self.s_ind+(self.batch_size-self.batch_count)
                    self.batch_count = self.batch_size
                    return np.asarray(images), np.asarray(labels).astype(int)
            self.epoch += 1
            self.index = self._make_index()
            self.s_ind = 0

    def augment(self, image):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring and np.random.randint(0, 2):
            image = np.fliplr(image)
        if self.rotation:
            image = np.rot90(image, np.random.randint(0, 4))
        return image

    def current_epoch(self):
        #TODO: return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.names.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        output = self.next()
        side = int(np.ceil(np.sqrt(self.batch_size)))
        f, axarr = plt.subplots(side, side, figsize=(2*side, 2*side))
        for i, image in enumerate(output[0]):
            axarr[i % side, i // side].imshow(image)
            axarr[i % side, i // side].set_title(self.class_name(output[1][i]), fontsize=10)
            axarr[i % side, i // side].axis("off")
        plt.show()
