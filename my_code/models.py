import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from my_code.model_tools import *


class Models:
    """
    主要从attention、语义分割、自编码器、GAN模型中寻找灵感
    model: self-attention / conv1d / mlp / gan / autoencoder
    """
    def __init__(self, m, constraint, channels):
        self.m, self.constraint = m, constraint
        self.channels = channels
        self.encoder = None
        self.decoder = None
        self.mapping_model = None

    def get_model(self, model_name, snr, input_shape, ofdm_outshape, ofdm_model):
        if model_name == 'mlp':
            model = MLP(m=self.m, constraint=self.constraint, ofdm_outshape=ofdm_outshape,
                        channels=self.channels, ofdm_model=ofdm_model, input_shape=input_shape)
            self.encoder, self.decoder, self.mapper = model.get_model(snr)
        elif model_name == 'attention':
            pass
        elif model_name == 'conv1d':
            #  input_shape: (batch_size, num_symbols, m)
            self.encoder = Conv1dEncoder(input_shape, self.channels)
            self.mapper = Mapper(input_shape, ofdm_model, snr)
            self.decoder = Conv1dDecoder(input_shape[1], self.m, self.channels)
        elif model_name == 'gan':
            pass
        elif model_name == 'autoencoder':
            pass
        else:
            raise Exception('No such model !')
        return self.encoder, self.decoder, self.mapper


class MLP:
    def __init__(self, m, constraint, channels, input_shape, ofdm_outshape, ofdm_model):
        self.layers_num = 4 if m < 4 else m
        self.m = m
        self.channels = channels
        self.constraint = constraint
        self.ofdm_model = ofdm_model
        self.input_shape = input_shape
        self.ofdm_outshape = ofdm_outshape
        self.__encoder__layers = [layers.Dense(2**i, activation='elu') for i in range(self.layers_num, 1, -1)]
        self.__encoder_out = layers.Dense(self.channels, activation='elu', name='encoder_out')
        self.__decoder_layers = [layers.Dense(2 ** i, activation='elu') for i in range(2, self.layers_num + 1, 1)]
        self.__decoder_out = layers.Dense(self.m, activation='sigmoid')
        if constraint == 'pow':
            self.__normalize_layer = PowerNormalize(name='normalize')
        elif constraint == 'amp':
            self.__normalize_layer = AmplitudeNormalize(name='normalize')
        else:
            pass
        if ofdm_model:
            self.__ofdm_layer = OFDMModulation(self.input_shape, name='ofdm')
            self.__de_ofdm_layer = OFDMDeModulation(self.ofdm_outshape, name='deofdm')

    def mapper(self, snr):
        noise_layer = self.channel(snr)
        mapper = keras.Sequential([layers.InputLayer(input_shape=(self.m,))], name="mapper")
        for layer in self.__encoder__layers:
            mapper.add(layer)
        mapper.add(self.__encoder_out)
        if self.constraint is not 'none':
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
        decoder = keras.Sequential([layers.InputLayer(input_shape=(self.channels,))], name="decoder")
        for layer in self.__decoder_layers:
            decoder.add(layer)
        decoder.add(self.__decoder_out)
        decoder.summary()
        return decoder

    def channel(self, snr):
        noise_layer = MyGaussianNoise(snr=snr, nbps=self.m, num_syms=self.input_shape, ofdm_model=self.ofdm_model, name='noise_layer')
        return noise_layer

    def get_model(self, snr):
        """
        mapper: 输出的是接收到的信号
        encoder：输出的是比特到符号映射的信号
        decoder：输出的是恢复出来的信号
        """
        mapper = self.mapper(snr)
        decoder = self.decoder()
        encoder = keras.Model(inputs=mapper.inputs,
                              outputs=mapper.get_layer(name='normalize').output if self.constraint is not 'none'
                              else mapper.get_layer(name='encoder_out').output, name='encoder')
        encoder.summary()
        return encoder, decoder, mapper


class Conv1dEncoder(keras.Model):
    def __init__(self, input_dim, channels):
        #  input_dim: (batch_size, time_step, length)-->(batch_size, num_symbols, m)
        super(Conv1dEncoder, self).__init__()
        self.conv1d1 = layers.Conv1D(2*channels, 3, padding='same', activation='elu', input_shape=input_dim[1:], name='cnv1')
        self.conv1d2 = layers.Conv1D(channels, 3, padding='same', activation='elu', name='cnv2')

    def call(self, inputs):
        out = self.conv1d2(self.conv1d1(inputs))
        return tf.reshape(out, (-1, 2))

    def get_config(self):
        return super(Conv1dEncoder, self).get_config()


class Conv1dDecoder(keras.Model):
    def __init__(self, syms, m, channels):
        #  input_dim: batch_size, num_symbols, channels
        super(Conv1dDecoder, self).__init__()
        self.in_dim = (-1, syms, channels)
        self.conv1d1 = keras.layers.Conv1D(channels, 3, padding='same', activation='elu', input_shape=self.in_dim[1:])
        self.conv1d2 = keras.layers.Conv1D(2*channels, 3, padding='same', activation='elu')
        self.conv1d3 = keras.layers.Conv1D(m, 3, padding='same', activation='elu')

    def call(self, inputs):
        inputs = tf.reshape(inputs, self.in_dim)  # (-1,236,2)
        o1 = self.conv1d1(inputs)
        return self.conv1d3(self.conv1d2(o1))

    def get_config(self):
        return {'in_dim': self.in_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Mapper(keras.Model):
    def __init__(self, input_shape, ofdm_model, snr):
        super(Mapper, self).__init__()
        num_syms = input_shape[0] * input_shape[1]
        m = input_shape[2]
        self.ofdm_model = ofdm_model
        ofdm_outshape = num_syms // OFDMParameters.fft_num.value * OFDMParameters.ofdm_syms.value
        self.encoder = Conv1dEncoder(input_dim=input_shape, channels=2)
        self.ofdm_ifft = OFDMModulation(num_syms)
        self.noise_layer = MyGaussianNoise(snr=snr, ofdm_model=ofdm_model, num_syms=num_syms, nbps=m)
        self.ofdm_fft = OFDMDeModulation(ofdm_outshape)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        if self.ofdm_model:
            encoded = self.ofdm_ifft(encoded)
        out = self.noise_layer(encoded)
        if self.ofdm_model:
            out = self.ofdm_fft(out)
        return out

    def get_config(self):
        config = super(Mapper, self).get_config()
        config.update({'ofdm_model': self.ofdm_model})
        return config


class Conv1dAE(keras.Model):
    """
    模型管理
    """
    def __init__(self, input_dim, input_shape, ofdm_model, syms,
                 m, snr, channels, name='Conv1dAE', **kwargs):
        super(Conv1dAE, self).__init__(name, **kwargs)
        self.encoder = Conv1dEncoder(input_dim, channels)
        self.mapper = Mapper(input_shape, ofdm_model, snr)
        self.decoder = Conv1dDecoder(syms, m, channels)

    def call(self, inputs):
        return self.decoder(self.mapper(inputs))


def main():
    ###################  mlp  ###############

    # train_data = np.random.randint(0, 2, (3762, 3))
    # M = Models(m=3, constraint='amp', channels=2)
    # #  encoder:用于validation，mapper用于生成星座图便于观察
    # encoder, decoder, mapper\
    #     = M.get_model(model_name='mlp', snr=23, input_shape=3762, ofdm_outshape=3762//64*80, ofdm_model=True)
    # mapping_out = mapper(train_data)
    # trans_out = encoder(train_data)
    # recover_out = decoder(trans_out)


    ###################  conv1d  #############

    input_shape = (16, 236, 3)
    x = tf.random.normal(input_shape)
    M = Models(m=3, constraint='amp', channels=2)
    encoder, decoder, mapper \
             = M.get_model(model_name='conv1d', snr=23, input_shape=input_shape,
                           ofdm_outshape=input_shape[0]*input_shape[1]//OFDMParameters.fft_num.value * OFDMParameters.ofdm_syms.value,
                           ofdm_model=0)
    y = mapper(x)
    print(y.shape)
    z = decoder(y)
    print(z.shape)
    y_ = encoder(x)
    encoder.save("../test/1")
    encoder_new = keras.models.load_model("../test/1")
    print(y_)
    print(encoder_new(x))
    print('finish!')




if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()
