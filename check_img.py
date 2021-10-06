import cv2
import numpy as np
import os
import pickle

def unpickle(file):

    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def main(cifar10_data_dir):
    for i in range(1, 6):
        train_data_file = os.path.join(cifar10_data_dir, 'data_batch_' + str(i))
        print(train_data_file)
        data = unpickle(train_data_file)
        print('unpickle done')
        for j in range(10):
            img = np.reshape(data[b'data'][j], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_name = 'train/' + str(data[b'labels'][j]) + '_' + str(j + (i - 1) * 10000) + '.jpg'
            cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)

    test_data_file = os.path.join(cifar10_data_dir, 'test_batch')
    data = unpickle(test_data_file)
    for i in range(10):
        img = np.reshape(data[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_name = 'test/' + str(data[b'labels'][i]) + '_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)


if __name__ == "__main__":
    main('/home/xiaoyang/Dev/AI6103-assignment/data/cifar-10-batches-py')

