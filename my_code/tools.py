import os
import pickle
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from tensorflow import keras
import json
from typing import NamedTuple
from enum import Enum


def json_serializable(cls):
    def as_dict(self):
        yield {name: value for name, value in zip(
            self._fields,
            iter(super(cls, self).__iter__()))}
    cls.__iter__ = as_dict
    return cls


GH_PATH = namedtuple('gh_path', 'g_path, h_path')
Data = namedtuple('data', 'train_data, train_label, test_data, test_label')
Result_save_path = namedtuple('result_save_path',
                              'mapper_train_pre, mapper_test_pre, decoder_train_pre, decoder_test_pre, train_bits, test_bits')
Model_save_path = namedtuple('model_save_path', 'encoder_save_path, decoder_save_path, mapper_save_path')


@json_serializable
class History(NamedTuple):
    epoch: list
    loss: list
    val_loss: list
    accuracy: list
    val_accuracy: list
    papr: list
    val_papr: list
    # namedtuple('hist', 'epoch, loss, val_loss, accuracy, val_accuracy, papr, val_papr')


def load_mat(path):
    data = sio.loadmat(path)
    print("data structure:\n")
    print([key for key in data.keys()])
    return data


def plot(hist:History, start):
    plt.figure()
    plt.plot(hist.epoch[start:], hist.loss[start:], label="loss")
    plt.plot(hist.epoch[start:], hist.val_loss[start:], label="val_loss")
    plt.legend()
    plt.figure()
    plt.plot(hist.epoch[start:], hist.papr[start:], label="papr")
    plt.plot(hist.epoch[start:], hist.val_papr[start:], label="val_papr")
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


def FindLCM(lcm, x=48, y=256):
    val = lcm
    while not (val % x == 0 and val % y == 0):
        val += 1
    return val


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
        if model_name == 'conv1d':
            data = data.reshape(-1)
            # padding = np.zeros(OFDMParameters.fft_num.value//16*self.__m -
            #                          data.shape[0]%(OFDMParameters.fft_num.value//16*self.__m),)
            # data0 = np.concatenate((data, padding))
            padding = np.zeros(FindLCM(data.shape[0])-data.shape[0],)  # debug 16*3 和 256的公倍数
            data = np.concatenate((data, padding))
            return data.reshape(OFDMParameters.fft_num.value//16, -1, self.__m).astype('float32')  # batch, k, m
        padding = np.zeros((OFDMParameters.fft_num.value - data.shape[0]%OFDMParameters.fft_num.value, self.__m))
        return np.concatenate((data, padding), axis=0)

    def decode(self, data, model_name='mlp'):
        if model_name == 'mlp':
            return data.reshape(-1, self.__g.shape[1])
        if model_name == 'conv1d':
            pass


    def __bits_process(self, bits: np.ndarray):
        length = bits.shape[0]
        if length % self.__shape[0] == 0:
            return bits.reshape(length // self.__shape[0], self.__shape[0])
        else:
            return np.concatenate((bits, np.zeros(self.__shape[0] - length % self.__shape[0]))) \
                .reshape(length // self.__shape[0] + 1, self.__shape[0])


def model_load(model_path: Model_save_path):
    mapper = keras.models.load_model(model_path.mapper_save_path)
    encoder = keras.models.load_model(model_path.encoder_save_path)
    decoder = keras.models.load_model(model_path.decoder_save_path)
    return encoder, decoder, mapper


def history_to_dict(hist: History):
    hist_dict = {}
    for filed_name, filed_data in zip(hist._fields, hist):
        hist_dict[filed_name] = filed_data
    return hist_dict


class Saver:
    @classmethod
    def save_model(cls, encoder, decoder, mapper, x_train, x_test,
                   model_save_path: Model_save_path, result_save_path:Result_save_path):
        root = model_save_path[0].split('/')
        mapper_train_pre = mapper(x_train)
        mapper_test_pre = mapper(x_test)
        decoder_train_pre = decoder(mapper(x_train))
        decoder_test_pre = decoder(mapper(x_test))
        sio.savemat(result_save_path.mapper_train_pre, {'mapper_train_result': mapper_train_pre.numpy()})
        sio.savemat(result_save_path.mapper_test_pre, {'mapper_test_result': mapper_test_pre.numpy()})
        sio.savemat(result_save_path.decoder_train_pre, {'decoder_train_result': decoder_train_pre.numpy()})
        sio.savemat(result_save_path.decoder_test_pre, {'decoder_test_result': decoder_test_pre.numpy()})
        make_dirs(model_save_path, root[0]+'/'+root[1]+'/')
        keras.models.save_model(encoder, model_save_path.encoder_save_path)
        keras.models.save_model(decoder, model_save_path.decoder_save_path)
        keras.models.save_model(mapper, model_save_path.mapper_save_path)

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

    @classmethod
    def save_history(cls, path, hist: History):
        with open(path, 'w') as f:
            json.dump(hist, f)


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

    fft_num = 256
    guard_interval = fft_num//8
    ofdm_syms = fft_num + guard_interval


if __name__ == "__main__":
    # model_save_path = Model_save_path(r'../my_model/mlp_encoder', r'../my_model/mlp_decoder',
    #                                   r'../my_model/mlp_mapper')
    # target_path = '../my_model/'
    # make_dirs(model_save_path, target_path)
    # history = History([1, 2], [2], [3, 4], [4], [5], [6], [7])
    # Saver.save_history("../data/history.json", history)
    print(FindLCM(11286))
    print("finish")

