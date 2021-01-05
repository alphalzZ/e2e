import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model_tools import *


class Models:
    """
    主要从attention、语义分割、自编码器、GAN模型中寻找灵感
    model: self-attention / conv1d / mlp / gan / autoencoder
    """
    def __init__(self, m, k, constraint, channels):
        self.m, self.k, self.constraint = m, k, constraint
        self.channels = channels
        self.encoder = None
        self.decoder = None
        self.mapping_model = None

    def get_model(self, model_name, snr, input_shape, ofdm_outshape, ofdm_model):
        if model_name == 'mlp':
            model = MLP(m=self.m, k=self.k, constraint=self.constraint, ofdm_outshape=ofdm_outshape,
                        channels=self.channels, ofdm_model=ofdm_model, input_shape=input_shape)
            self.encoder, self.decoder, self.mapping_model = model.get_model(snr)
        elif model_name == 'attention':
            pass
        elif model_name == 'conv1d':
            pass
        elif model_name == 'gan':
            pass
        elif model_name == 'autoencoder':
            pass
        else:
            raise Exception('No such model !')
        return self.encoder, self.decoder, self.mapping_model


class MLP:
    def __init__(self, m, k, constraint, channels, input_shape, ofdm_outshape, ofdm_model):
        self.m, self.k = m, k
        self.channels = channels
        self.constraint = constraint
        self.ofdm_model = ofdm_model
        self.input_shape = input_shape
        self.ofdm_outshape = ofdm_outshape

    def encoder(self, snr):
        model_layers = [layers.Dense(2**i, activation='elu') for i in range(self.m, 1, -1)]
        model_out = layers.Dense(self.channels, activation='elu')
        ofdm_layer = OFDMModulation(self.input_shape, name='ofdm')
        noise_layer = GaussianNoise(snr=snr, num_syms=self.input_shape,
                                    ofdm_model=self.ofdm_model, num_sym=80, nbps=self.m, name='noise')
        de_ofdm_layer = OFDMDeModulation(self.ofdm_outshape, name='deofdm')
        if self.constraint == 'pow':
            normalize_layer = PowerNormalize()
        elif self.constraint == 'amp':
            normalize_layer = MappingLayer()
        else:
            normalize_layer = layers.BatchNormalization()

        encode_model = keras.Sequential([layers.Input(shape=(self.m,),)], name="encoder")
        for layer in model_layers:
            encode_model.add(layer)
        encode_model.add(model_out)
        encode_model.add(normalize_layer)
        if self.ofdm_model:
            encode_model.add(ofdm_layer)  # 输入符号个数设置为64的整数倍
        encode_model.add(noise_layer)
        if self.ofdm_model:
            encode_model.add(de_ofdm_layer)
        encode_model.summary()
        mapping_model = keras.Model(inputs=encode_model.inputs,
                                    outputs=ofdm_layer.output if self.ofdm_model else normalize_layer.output)
        return encode_model, mapping_model

    def decoder(self):
        model_layers = [layers.Dense(2**i, activation='elu') for i in range(2, self.m+1, 1)]
        model_out = layers.Dense(self.m, activation='sigmoid')
        decode_model = keras.Sequential([keras.Input(shape=(2,))], name="decoder")
        for layer in model_layers:
            decode_model.add(layer)
        decode_model.add(model_out)
        decode_model.summary()
        return decode_model

    def get_model(self, snr):
        encoder, mapping_model = self.encoder(snr)
        decoder = self.decoder()
        return encoder, decoder, mapping_model


def main():
    N, m, k = 128, 4, 75  # 分组个数， 每符号比特数， 符号个数
    train_data = np.random.randint(0, 2, (N*k, m))
    M = Models(m=m, k=k, constraint='amp', channels=2)
    #  mapping_model:用于validation，mapping_model_constellation用于生成星座图便于观察
    encoder, decoder, mapping_model\
        = M.get_model(model_name='mlp', snr=23, input_shape=N*k, ofdm_outshape=128*75//64*80, ofdm_model=True)
    mapping_out = mapping_model(train_data)
    trans_out = encoder(train_data)
    recover_out = decoder(trans_out)
    print('finish!')


if __name__ == "__main__":
    main()
