import argparse
import numpy as np
from termcolor import cprint
import time

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import model_from_json
from keras.models import Sequential
from keras.utils import plot_model


EPOCHS = 100
BATCH_SIZE = 512 * 3


class Cifar_classifier():
    def __init__(self, dir_path):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = \
            cifar10.load_data()

    def make_model_from_pre_trained(self, pre_trained_model_path):
        with open(pre_trained_model_path + ".json", "rt")as f:
            json_model = f.read()
        self.model = model_from_json(json_model)
        self.model.compile(loss="mean_squared_error",
                           optimizer="adam",
                           metrics=["accuracy"])
        self.model.load_weights(pre_trained_model_path + ".h5")
        # print(self.model.summary())

    def make_model(self, pre_trained_model_path=None):
        self.model = Sequential()

        self.model.add(Conv2D(32, 3, input_shape=self.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train,
                       self.dtrain.target.values,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       verbose=1)

    def evaluate(self):
        pred = self.model.predict(self.X_valid, verbose=1)
        pred = self.target_scaler.inverse_transform(pred)
        pred = np.exp(pred) - 1

        y_true = np.array(self.dvalid.price.values)
        y_pred = pred[:, 0]
        v_rmsle = rmsle(y_true, y_pred)
        cprint("RMSLE error on dev test: " + str(v_rmsle), "green")

    def predict(self, input_data):
        pred = self.model.predict(input_data, verbose=1, batch_size=BATCH_SIZE)
        pred = self.target_scaler.inverse_transform(pred)
        pred = np.exp(pred) - 1
        return pred

    def save_model(self, checkpoint_path):
        self.model.save_weights(checkpoint_path + ".h5")
        with open(checkpoint_path + ".json", "w") as f:
            f.write(self.model.to_json())


def get_callbacks(checkpoint_path, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(checkpoint_path, save_best_only=True)
    return [es, msave]


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return (np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean()) ** 0.5


def argparser():
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument("-m", "--mode",
                        default=None,
                        nargs="?",
                        help="train or predict")
    parser.add_argument("-i", "--input_dir_path",
                        default="./data/",
                        nargs="?",
                        help="input data path")
    parser.add_argument("-p", "--pre_trained_model_path",
                        default=None,
                        nargs="?",
                        help="to load checkpoint h5 file path")
    parser.add_argument("-c", "--checkpoint_path",
                        default="./data/save_test",
                        nargs="?",
                        help="checkpoint h5 file path")
    parser.add_argument("-s", "--submission_path",
                        default="./data/NNsubmission.csv",
                        nargs="?",
                        help="csv file path for submission")
    return parser.parse_args()


def time_measure(section, start, elapsed):
    lap = time.time() - start - elapsed
    elapsed = time.time() - start
    cprint("{:22}: {:10.2f}[sec]{:10.2f}[sec]".format(section, lap, elapsed),
           "blue")
    return elapsed


def main():
    start = time.time()
    args = argparser()
    print(args)
    mercari = Cifar_classifier(args.input_dir_path)
    elapsed = time_measure("load data", start, 0)
    mercari.handle_nan_process()
    elapsed = time_measure("handle nan", start, elapsed)
    mercari.label_encode()
    elapsed = time_measure("label encode data", start, elapsed)
    mercari.tokenize_seq_data()
    elapsed = time_measure("tokenize data", start, elapsed)
    mercari.define_max()
    mercari.arrange_target()
    mercari.get_keras_data_process()
    elapsed = time_measure("complete arrange data", start, elapsed)
    mercari.make_model()
    plot_model(mercari.model, to_file="./data/model.png", show_shapes=True)
    elapsed = time_measure("make model", start, elapsed)
    mercari.train_model()
    elapsed = time_measure("train model", start, elapsed)
    mercari.save_model(args.checkpoint_path)
    elapsed = time_measure("save model", start, elapsed)
    mercari.evaluate()
    elapsed = time_measure("evaluation", start, elapsed)
    mercari.make_submission(args.submission_path)
    elapsed = time_measure("make submission", start, elapsed)
    K.clear_session()


if __name__ == "__main__":
    main()
