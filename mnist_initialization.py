from mnist import MNIST
import numpy as np

def load_data_mnist():
    mdata = MNIST('data')
    train_images, train_labels = mdata.load_training()
    test_images, test_labels = mdata.load_testing()
    #print(np.array(train_images).shape)

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)
