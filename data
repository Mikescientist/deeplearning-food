from __future__ import division, print_function, absolute_import
import urllib
import random
import os
import zipfile


def create_dataset():
    url = 'http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip'
    print('Downloading Images')
    urllib.urlretrieve(url, "../Python/images.zip")

    print('Extracting files')
    z = zipfile.ZipFile("../Python/images.zip") # change to your desired directory
    z.extractall()


# create directory for dataset in your desired directory

    datapath = r'../Python/data/'
    trainpath = r'../Python/data/train/'
    testpath = r'../Python/data/test/'
    sandpath1 = r'../Python/data/train/sandwiches/'
    sushpath1 = r'../Python/data/train/sushi/'
    sandpath2 = r'../Python/data/test/sandwiches/'
    sushpath2 = r'../Python/data/test/sushi/'


    dir_list = [datapath, trainpath, testpath, sandpath1, sandpath2, sushpath1, sushpath2]
    for pathname in dir_list:
        if not os.path.exists(pathname):
            os.makedirs(pathname)
    path1 = '../Python/sushi_or_sandwich/sandwich/'
    path2 = '../Python/sushi_or_sandwich/sushi/'
    X = os.listdir(path1)
    Y = os.listdir(path2)


# Randomize dataset

    random.shuffle(X)
    random.shuffle(Y)


# Make a cross validation split of ~63/37 to create training and test set

    training_no = 302
    for i in range(0, training_no):
        with open(path1 + X[i], 'rb') as f:
            img = f.read()
        with open(sandpath1 + X[i], 'wb') as f:
            f.write(img)
            f.close()
        with open(path2 + Y[i], 'rb') as f:
            img = f.read()
        with open(sushpath1 + Y[i], 'wb') as f:
            f.write(img)
            f.close()

    for i in range(training_no, len(X)):
        with open(path1 + X[i], 'rb') as f:
            img = f.read()
        with open(sandpath2 + X[i], 'wb') as f:
            f.write(img)
            f.close()
        with open(path2 + Y[i], 'rb') as f:
            img = f.read()
        with open(sushpath2 + Y[i], 'wb') as f:
            f.write(img)
            f.close()
