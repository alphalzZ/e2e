import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from code.model_tools import *


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
        self.ofdm_model = ofdm_model
        self.input_shape = input_shape
        self.ofdm_outshape = ofdm_outshape
        self.__encoder__layers = [layers.Dense(2**i, activation='elu') for i in range(self.m, 1, -1)]
        self.__encoder_out = layers.Dense(self.channels, activation='elu')
        self.__decoder_layers = [layers.Dense(2 ** i, activation='elu') for i in range(2, self.m + 1, 1)]
        self.__decoder_out = layers.Dense(self.m, activation='sigmoid')
        if constraint == 'pow':
            self.__normalize_layer = PowerNormalize(name='normalize')
        elif constraint == 'amp':
            self.__normalize_layer = MappingLayer(name='normalize')
        else:
            self.__normalize_layer = layers.BatchNormalization(name='normalize')
        if ofdm_model:
            self.__ofdm_layer = OFDMModulation(self.input_shape, name='ofdm')
            self.__de_ofdm_layer = OFDMDeModulation(self.ofdm_outshape, name='deofdm')

    def mapper(self, snr):
        noise_layer = self.channel(snr)
        mapper = keras.Sequential([layers.InputLayer(input_shape=(self.m,))], name="mapper")
        for layer in self.__encoder__layers:
            mapper.add(layer)
        mapper.add(self.__encoder_out)
        mapper.add(self.__normalize_layer)
        if self.ofdm_model:
            mapper.add(self.__ofdm_layer)
            mapper.add(noise_layer)
            mapper.add(self.__de_ofdm_layer)
        else:
            mapper.add(noise_layer)
        mapper.summary()
        return mapper

    def decoder(self):
        decoder = keras.Sequential([layers.InputLayer(input_shape=(2,))], name="decoder")
        for layer in self.__decoder_layers:
            decoder.add(layer)
        decoder.add(self.__decoder_out)
        decoder.summary()
        return decoder

    def channel(self, snr):
        noise_layer = GaussianNoise(snr=snr, num_syms=self.input_shape,
                                    ofdm_model=self.ofdm_model, num_sym=80, nbps=self.m, name='noise')
        return noise_layer

    def get_model(self, snr):
        """
        mapper: 输出的是接收到的信号
        encoder：输出的是比特到符号映射的信号
        decoder：输出的是恢复出来的信号
        """
        mapper = self.mapper(snr)
        decoder = self.decoder()
        encoder = keras.Model(inputs=mapper.inputs, outputs=mapper.get_layer(name='normalize').output)
        mapper.summary()
        return encoder, decoder, mapper


def main():
    N, m, k = 128, 4, 75  # 分组个数， 每符号比特数， 符号个数
    train_data = np.random.randint(0, 2, (N*k, m))
    M = Models(m=m, k=k, constraint='amp', channels=2)
    #  encoder:用于validation，mapper用于生成星座图便于观察
    encoder, decoder, mapper\
        = M.get_model(model_name='mlp', snr=23, input_shape=N*k, ofdm_outshape=128*75//64*80, ofdm_model=True)
    mapping_out = mapper(train_data)
    trans_out = encoder(train_data)
    recover_out = decoder(trans_out)
    print('finish!')


if __name__ == "__main__":
    main()
