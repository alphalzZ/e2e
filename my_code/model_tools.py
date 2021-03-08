
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.keras import layers, losses, metrics
from my_code.tools import OFDMParameters
print('Tensorflow version:{}'.format(tf.__version__))


class MappingConstraint(losses.Loss):
    """
    Mapping constraint
    设计适合的约束项可以适当的减小PAPR
    """
    def __init__(self, mapping_method='none', name="mapping_constraint_loss"):
        super().__init__(name=name)
        self.mapping_method = mapping_method

    def call(self, y_pred, y_true):
        if self.mapping_method == 'papr':
            amplitude = tf.math.square(y_pred[:, 0]) + tf.math.square(y_pred[:, 1])
            papr_constraint = tf.math.reduce_max(amplitude)/tf.math.reduce_mean(amplitude)
            constraint = papr_constraint
        elif self.mapping_method == 'pow':
            constraint = tf.reduce_mean(tf.square(y_pred))
        else:
            constraint = 0.
        return constraint

    def get_config(self):
        return {"papr_method": self.papr_method}


class PAPRConstraint(losses.Loss):
    """num_syms: 符号个数总数"""
    def __init__(self, num_syms, snr, num_fft=OFDMParameters.fft_num.value,
                 num_guard=OFDMParameters.guard_interval.value, name="papr_constraint_loss"):
        super().__init__(name=name)
        self.num_fram = num_syms // num_fft
        self.num_ofdm_length = num_fft + num_guard
        self.__noise_layer = MyGaussianNoise(snr, ofdm_model=1, num_syms=num_syms)

    def call(self, y_true, y_pred):
        ofdm_shape = (self.num_fram, self.num_ofdm_length)  # （子载波数量，ofdm符号个数）
        ofdm_signal_iq = self.__noise_layer(y_pred)
        ofdm_signal_complex = tf.complex(ofdm_signal_iq[:, 0], ofdm_signal_iq[:, 1])
        ofdm_signal = tf.reshape(ofdm_signal_complex, ofdm_shape)
        power = tf.math.abs(ofdm_signal)**2
        mean_power = tf.reduce_mean(power, axis=1)
        max_power = tf.reduce_max(power, axis=1)
        papr_vector = max_power / mean_power
        papr = tf.reduce_mean(papr_vector)
        papr_log = 10*tf.experimental.numpy.log10(papr)
        return papr_log


class AmplitudeNormalize(layers.Layer):
    """
    amplitude constraint
    """
    def __init__(self, channel=2, **kwargs):
        super(AmplitudeNormalize, self).__init__(**kwargs)
        self.channel = channel

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return {"channel": self.channel}


class SelfAttention(layers.Layer):
    """
    reference: Attention is all you need
    """
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.softmax = layers.Softmax(axis=1)

    def build(self, input_shape):
        #  input_shape:(batch_size, m)
        self.w = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal',
                                 trainable=True, name='w')
        self.w_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal',
                                   trainable=True, name='w_q')
        self.w_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal',
                                   trainable=True, name='w_k')
        self.w_v = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='random_normal',
                                   trainable=True, name='w_v')

    def call(self, inputs):
        a = tf.matmul(inputs, self.w)  # (batch.m)
        q = tf.matmul(a, self.w_q)
        k = tf.matmul(a, self.w_k)
        v = tf.matmul(a, self.w_v)
        k_q = tf.matmul(tf.transpose(k), q)
        k_q_hat = self.softmax(k_q)
        out = tf.matmul(v, k_q_hat)
        return out

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        return config


class PowerNormalize(layers.Layer):
    """
    average power constraint
    TODO debug power constraint
    """
    def __init__(self, **kwargs):
        super(PowerNormalize, self).__init__(**kwargs)

    def call(self, inputs):
        norm_factor = (1./tf.reduce_mean(tf.reduce_sum(tf.square(inputs), axis=1)))**0.5
        normalize_power = inputs*norm_factor
        return normalize_power

    def get_config(self):
        return super(PowerNormalize, self).get_config()


class MyGaussianNoise(layers.Layer):
    """
    Gaussian Noise Layer
    SNR(dB) = 20×log10(1/ noise_std)
    noise_std = mean_pow/10**(snr/20)
    """
    def __init__(self, snr, ofdm_model, num_syms, num_sym=OFDMParameters.ofdm_syms.value, nbps=4, **kwargs):
        """num_sym:一个ofdm信号的长度"""
        super(MyGaussianNoise, self).__init__(**kwargs)
        self.snr = snr
        self.ofdm_model = ofdm_model
        self.num_sym = num_sym  # 一帧ofdm信号符号个数
        self.num_syms = num_syms  # 符号个数
        self.nbps = nbps

    def call(self, inputs):
        """如何给ofdm 信号加噪声"""
        num_fram = self.num_syms/self.num_sym
        if self.ofdm_model:
            complex_signal = tf.complex(inputs[:, 0], inputs[:, 1])
            mean_pow = tf.matmul(complex_signal[None, :], tf.math.conj(complex_signal[:, None]))/self.num_sym/num_fram
            snrs = self.snr + 10.*np.log10(float(self.nbps))
            noise_std = tf.sqrt(10.**(-snrs/10.) * tf.math.real(mean_pow)/2.)
        else:  # TODO debug add noise
            mean_pow = tf.reduce_mean(tf.reduce_sum(tf.square(inputs), axis=1))**0.5
            noise_std = (mean_pow/(10.**(2.*0.834*self.snr/10.)))**0.5  # np.sqrt(mean_pow/(2*R*EbNo))? R = Gr/Gc = 0.834
        noise = tf.random.normal(shape=array_ops.shape(inputs), mean=0., stddev=noise_std)
        return inputs + noise


class OFDMModulation(layers.Layer):
    def __init__(self, num_syms, num_gard=OFDMParameters.guard_interval.value,
                 num_fft=OFDMParameters.fft_num.value, **kwargs):
        """保护间隔个数， fft个数， 每一帧ofdm符号个数等于num_gard+num_fft， 帧数等于符号个数除以num_fft"""
        #  TODO 子载波个数 <= ifft点数
        super(OFDMModulation, self).__init__(**kwargs)
        self.num_gard, self.num_fft = num_gard, num_fft
        self.num_syms = num_syms
        self.prbatchnorm = PRBatchnorm(True, num_syms//num_fft)
        # self.prbatchnorm = AttentionBatchnorm()

    def call(self, inputs):  # (num_symbols, 2)
        num_fram = self.num_syms//self.num_fft
        X = tf.complex(inputs[:, 0], inputs[:, 1])
        X = tf.reshape(X[:, None], [num_fram, -1])
        x = tf.signal.ifft(X)
        x = tf.concat((x[:, self.num_fft - self.num_gard:self.num_fft], x), axis=1)
        x = self.prbatchnorm(x)  # (frams,syms)
        x = tf.reshape(x, [-1, 1])
        x = tf.concat((tf.math.real(x), tf.math.imag(x)), axis=1)
        return x

    def get_config(self):
        config = super(OFDMModulation, self).get_config()
        config.update({'num_gard':self.num_gard, 'num_fft': self.num_fft, 'num_syms':self.num_syms})
        return config


class OFDMDeModulation(layers.Layer):
    def __init__(self, ofdm_outshape, num_gard=OFDMParameters.guard_interval.value,
                 num_fft=OFDMParameters.fft_num.value, **kwargs):
        """保护间隔个数， fft个数， 每一帧ofdm符号个数等于num_gard+num_fft， 帧数等于符号个数除以num_fft"""
        super(OFDMDeModulation, self).__init__(**kwargs)
        self.num_gard, self.num_fft = num_gard, num_fft
        self.num_sym = num_gard + num_fft
        self.ofdm_outshape = ofdm_outshape
        self.prbatchnorm = PRBatchnorm(True, ofdm_outshape // self.num_sym)

    def call(self, inputs):
        num_fram = self.ofdm_outshape // self.num_sym
        Y = tf.complex(inputs[:, 0], inputs[:, 1])
        Y = tf.reshape(Y, (num_fram, -1))
        Y = self.prbatchnorm(Y)
        y = tf.signal.fft(Y[:, self.num_gard:self.num_sym])
        y = tf.reshape(y, (-1, 1))
        outputs = tf.concat((tf.math.real(y), tf.math.imag(y)), axis=1)
        return outputs

    def get_config(self):
        config = super(OFDMDeModulation, self).get_config()
        config.update({'num_gard':self.num_gard, 'num_fft': self.num_fft,
                       'num_sym': self.num_sym,'ofdm_outshape':self.ofdm_outshape})
        return config


class PRBatchnorm(layers.Layer):
    """
    A Novel PAPR Reduction Scheme for OFDM System based on Deep Learning
    """
    def __init__(self, trainable, num_fram, **kwargs):
        super(PRBatchnorm, self).__init__(**kwargs)
        self.trainable = trainable
        self.num_fram = num_fram

    # def build(self, input_shape):
    #     """原文中是两个标量, 此处是每一个子载波对应一个缩放因子"""
    #     self.gama = self.add_weight(
    #         shape=(self.num_fram, 1, 1),
    #         initializer=tf.initializers.constant(1.),
    #         trainable=self.trainable,
    #         name='gama'
    #     )
    #     self.beta = self.add_weight(
    #         shape=(self.num_fram, 1, 1), initializer=tf.initializers.constant(0.001), trainable=self.trainable,
    #         name='beta'
    #     )

    def build(self, input_shape):
        """原文中是两个标量, 此处是每一个子载波对应一个缩放因子"""
        shape = (self.num_fram, 1, 2)
        self.gama = tf.Variable(
            initial_value=tf.random.normal(shape, mean=1, stddev=0.001),
            trainable=self.trainable,
            name='gama'
        )
        self.beta = tf.Variable(
            initial_value=0.001*tf.random.normal(shape, stddev=0.001),
            trainable=self.trainable,
            name='beta'
        )
        # self.gama = tf.Variable(
        #     initial_value=tf.random.normal((1, 1), mean=1, stddev=0.001),
        #     trainable=self.trainable,
        #     name='gama'
        # )
        # self.beta = tf.Variable(
        #     initial_value=0.001 * tf.random.normal((1, 1), stddev=0.001),
        #     trainable=self.trainable,
        #     name='beta'
        # )

    def call(self, inputs, training=None):
        """inputs:复信号，（num_fram, num_syms）"""
        inputs = tf.concat((tf.math.real(inputs)[:, :, None], tf.math.imag(inputs)[:, :, None]), axis=2)
        mean = tf.reduce_mean(inputs, axis=1)  # 每一个子载波的均值
        inputs = tf.subtract(inputs, mean[:, None, :])
        sigma = tf.math.reduce_variance(inputs, axis=1) # 每一个子载波的方差
        x = tf.add(tf.multiply(self.gama, inputs)/tf.sqrt(tf.add(sigma[:, None, :], tf.constant(0.001))), self.beta)
        return tf.complex(x[:, :, 0], x[:, :, 1])

    def get_config(self):
        config = super(PRBatchnorm, self).get_config()
        config.update({'trainable': self.trainable, 'num_fram': self.num_fram})
        return config


class PRBatchnorm_2(layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super(PRBatchnorm_2, self).__init__(**kwargs)
        self.trainable = trainable

    def build(self, input_shape):
        """原文中是两个标量, 此处是每一个子载波对应一个缩放因子"""
        self.gama = tf.Variable(
            initial_value=tf.random.normal((1, 1), mean=0.1, stddev=0.001),
            trainable=self.trainable,
            name='gama'
        )
        self.beta = tf.Variable(
            initial_value=0.001 * tf.random.normal((1, 1), stddev=0.01),
            trainable=self.trainable,
            name='beta'
        )

    def call(self, inputs, training=None):
        """inputs:复信号，（num_signals, m）"""
        mean = tf.reduce_mean(inputs, axis=0)
        inputs = tf.subtract(inputs, mean[None, :])
        sigma = tf.math.reduce_variance(inputs, axis=0)
        x = tf.add(tf.multiply(self.gama, inputs) / tf.sqrt(tf.add(sigma[None, :], tf.constant(0.001))), self.beta)
        return x

    def get_config(self):
        config = super(PRBatchnorm_2, self).get_config()
        config.update({'trainable': self.trainable})
        return config


class AttentionBatchnorm(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionBatchnorm, self).__init__(**kwargs)
        self.attention = SelfAttention()

    def call(self, inputs, training=None):
        """inputs:复信号，（num_fram, num_syms）"""
        inputs = tf.concat((tf.math.real(inputs)[:, :, None], tf.math.imag(inputs)[:, :, None]), axis=2)
        out = []
        for i in range(inputs.shape[0]):
            out.append(self.attention(inputs[i, :, :]))
        out = tf.convert_to_tensor(out)
        return tf.complex(out[:, :, 0], out[:, :, 1])


def clipping_functions(x):
    return 0.1*keras.activations.relu(x+5)-0.1*keras.activations.relu(x-5)


def clipping(x):
    sigma = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.abs(x))))
    CL = 0.3*sigma
    x_clipped = x
    clipped_idx = tf.cast(tf.math.abs(x_clipped) > CL, tf.float32)
    z = x_clipped*clipped_idx
    z = tf.math.divide((z*CL), tf.math.abs(z)+1e-4)
    return z


if __name__ == "__main__":
    Input = np.random.randn(32, 3)
    atten = SelfAttention()
    out = atten(Input)
    mapping_pre = np.random.randn(512, 2)
    # complex_signal = tf.complex(tf.random.uniform((2, 32)), tf.random.uniform((2, 32)))
    # prbatchnorm = PRBatchnorm(True, 2)
    # out0 = prbatchnorm(complex_signal)
    # papr_loss = MappingConstraint()
    # loss = papr_loss(y_pred=mapping_pre, y_true=mapping_pre)
    # m = MappingConstraint()
    # out = m(mapping_pre, mapping_pre)
    # signal = np.random.randn(10, 2)
    # Noise = MyGaussianNoise(10, ofdm_model=False, num_syms=128)
    # out2 = Noise(signal)
    # PN = PowerNormalize()
    # out3 = PN(np.array([[1.,2.], [3.,4.], [5.,6.]]))
    ofdm = OFDMModulation(num_sym=512)
    ofdm_out = ofdm(mapping_pre)
    deofdm = OFDMDeModulation(ofdm_outshape=512//256*288)
    deofdm_out = deofdm(ofdm_out)
    papr_constraint = PAPRConstraint(512, 23)
    out4 = papr_constraint(mapping_pre, mapping_pre)
    print("finish!")
