import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_batch_1 = unpickle("data/cifar-10-batches-py/data_batch_1")
data_batch_1
data_batch_1[b"data"].shape
len(unpickle("data/cifar-10-batches-py/data_batch_1")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_2")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_3")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_4")[b"labels"])
len(unpickle("data/cifar-10-batches-py/data_batch_5")[b"labels"])
