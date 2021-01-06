import os
import pickle
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from tensorflow import keras
from enum import Enum


GH_PATH = namedtuple('gh_path', 'g_path, h_path')
Data = namedtuple('data', 'train_data, train_label, test_data, test_label')
History = namedtuple('hist', 'epoch, loss, val_loss')
Result_save_path = namedtuple('result_save_path',
                              'mapper_train_pre, mapper_test_pre, decoder_train_pre, decoder_test_pre, train_bits, test_bits')
Model_save_path = namedtuple('model_save_path', 'encoder_save_path, decoder_save_path, mapper_save_path')


def load_mat(path):
    data = sio.loadmat(path)
    print("data structure:\n")
    print([key for key in data.keys()])
    return data


def plot(hist, start):
    plt.figure()
    plt.plot(hist.epoch[start:], hist.loss[start:], label="loss")
    plt.plot(hist.epoch[start:], hist.val_loss[start:], label="val_loss")
    plt.legend()
    plt.show()


def qam_process(m, bits_matrix: np.ndarray):
    """
    将矩阵末尾补一个零矩阵
    """
    bits_per_symbol = m
    shape = bits_matrix.shape
    if shape[1] % bits_per_symbol != 0:
        columns = int(bits_per_symbol - shape[1] % bits_per_symbol)
        padding = np.zeros((shape[0], columns))
        bits_matrix = np.concatenate((bits_matrix, padding), axis=1)
    return bits_matrix


class LDPCEncode:
    def __init__(self, g, m, bit_nums):
        self.__g = g
        self.__m = m
        self.__shape = g.shape
        self.__bit_nums = bit_nums // g.shape[0] * g.shape[0]
        self.bits = None

    def encode(self, bits=None, model_name='mlp'):
        self.bits = np.random.randint(0, 2, (self.__bit_nums // self.__g.shape[0], self.__g.shape[0]))  # 生成二维bit矩阵
        print("bits shape: {}".format(self.bits.shape))
        if bits is not None:
            self.bits = self.__bits_process(bits)
        data = qam_process(self.__m, np.mod(np.matmul(self.bits, self.__g), 2))
        if model_name == 'mlp':
            data = data.reshape(-1, self.__m).astype('float32')
        return data

    def decode(self, data, model_name='mlp'):
        if model_name == 'mlp':
            return data.reshape(-1, self.__g.shape[1])

    def __bits_process(self, bits: np.ndarray):
        length = bits.shape[0]
        if length % self.__shape[0] == 0:
            return bits.reshape(length // self.__shape[0], self.__shape[0])
        else:
            return np.concatenate((bits, np.zeros(self.__shape[0] - length % self.__shape[0]))) \
                .reshape(length // self.__shape[0] + 1, self.__shape[0])


def model_load(model_path: Model_save_path):
    mapper = tf.saved_model.load(model_path.mapper_save_path)
    encoder = tf.saved_model.load(model_path.encoder_save_path)
    decoder = tf.saved_model.load(model_path.decoder_save_path)
    return encoder, decoder, mapper


class Saver:
    @classmethod
    def save_model(cls, encoder, decoder, mapper, x_train, x_test,
                   model_save_path: Model_save_path, result_save_path:Result_save_path):
        mapper_train_pre = mapper(x_train)  # 经过信道加完噪声
        mapper_test_pre = mapper(x_test)
        decoder_train_pre = decoder(mapper(x_train))
        decoder_test_pre = decoder(mapper(x_test))
        sio.savemat(result_save_path.mapper_train_pre, {'mapper_train_result': mapper_train_pre.numpy()})
        sio.savemat(result_save_path.mapper_test_pre, {'mapper_test_result': mapper_test_pre.numpy()})
        sio.savemat(result_save_path.decoder_train_pre, {'decoder_train_result': decoder_train_pre.numpy()})
        sio.savemat(result_save_path.decoder_test_pre, {'decoder_test_result': decoder_test_pre.numpy()})
        make_dirs(model_save_path, "../my_model/")
        tf.saved_model.save(encoder, model_save_path.encoder_save_path)
        tf.saved_model.save(decoder, model_save_path.decoder_save_path)
        tf.saved_model.save(mapper, model_save_path.mapper_save_path)

    @classmethod
    def save_result(cls, result_save_path: Result_save_path, train_bits, test_bits):
        sio.savemat(result_save_path.train_bits, {'train_bits': train_bits})
        sio.savemat(result_save_path.test_bits, {'test_bits': test_bits})

    @classmethod
    def save_class_element(cls, Object: object, name, path):
        element = getattr(Object, name)
        with open(path, 'wb') as f:
            pickle.dump(element, f)

    @classmethod
    def load_class_element(cls, path):
        with open(path, 'rb') as f:
            element = pickle.load(f)
        return element


def make_dirs(save_path, target_path):
    """
    save_path:全路径
    target_path:主路径
    """
    exist_path = os.listdir(target_path)
    paths = [path.replace(target_path, '') for path in save_path]
    for path in paths:
        if path not in exist_path:
            os.makedirs(target_path+path)


class OFDMParameters(Enum):
    guard_interval = 16
    fft_num = 64
    ofdm_syms = 80


if __name__ == "__main__":
    model_save_path = Model_save_path(r'./my_model/mlp_encoder_2', r'./my_model/mlp_decoder',
                                      r'./my_model/mlp_mapper')
    target_path = '../my_model/'
    make_dirs(model_save_path, target_path)