import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.keras import layers, losses
print('Tensorflow version:{}'.format(tf.__version__))


class PAPRConstraint(losses.Loss):
    """
    PAPR constraint
    设计适合的约束项可以适当的减小PAPR
    """
    def __init__(self, papr_method='none', name="papr_constraint_loss"):
        super().__init__(name=name)
        self.papr_method = papr_method

    def call(self, y_pred, y_true):
        constraint = 0.
        if self.papr_method == 'papr':
            amplitude = tf.math.square(y_pred[:, 0]) + tf.math.square(y_pred[:, 1])
            papr_constraint = tf.math.reduce_max(amplitude)/tf.math.reduce_mean(amplitude)
            constraint = papr_constraint
        elif self.papr_method == 'pow':
            constraint = tf.reduce_mean(tf.square(y_pred))
        return constraint

    def get_config(self):
        return {"papr_method": self.papr_method}


class MappingLayer(layers.Layer):
    """
    amplitude constraint
    """
    def __init__(self, channel=2, units=None, **kwargs):
        super(MappingLayer, self).__init__(**kwargs)
        self.units = units
        self.channel = channel

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return {"units": self.units, "channel": self.channel}


class PowerNormalize(layers.Layer):
    """
    average power constraint
    TODO debug power constraint
    """
    def __init__(self, units=None, **kwargs):
        super(PowerNormalize, self).__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        norm_factor = (1./tf.reduce_mean(tf.reduce_sum(tf.square(inputs), axis=1)))**0.5
        normalize_power = inputs*norm_factor
        return normalize_power

    def get_config(self):
        return {"units": self.units}


class GaussianNoise(layers.Layer):
    """
    Gaussian Noise Layer
    SNR(dB) = 20×log10(1/ noise_std)
    noise_std = mean_pow/10**(snr/20)
    """
    def __init__(self, snr, ofdm_model, num_syms, num_sym=80, nbps=4, **kwargs):
        """num_sym:一个ofdm信号的长度"""
        super(GaussianNoise, self).__init__(**kwargs)
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
            noise_std = mean_pow/(10.**(self.snr/20.))
        noise = tf.random.normal(shape=array_ops.shape(inputs), mean=0., stddev=noise_std)
        return inputs + noise

    def get_config(self):
        config = super(GaussianNoise, self).get_config()
        config.update({'snr': self.snr, 'ofdm_model': self.ofdm_model, 'num_sym':self.num_sym,
                       'num_syms': self.num_syms, 'nbps':self.nbps})
        return config


class OFDMModulation(layers.Layer):
    def __init__(self, num_sym, num_gard=16, num_fft=64, **kwargs):
        """保护间隔个数， fft个数， 每一帧ofdm符号个数等于num_gard+num_fft， 帧数等于符号个数除以num_fft"""
        super(OFDMModulation, self).__init__(**kwargs)
        self.num_gard, self.num_fft = num_gard, num_fft
        self.num_sym = num_sym

    def call(self, inputs):  # (num_symbols, 2)
        x_GI = []
        num_fram = self.num_sym//self.num_fft
        for i in range(num_fram):
            X = tf.complex(inputs[i*self.num_fft: (i+1)*self.num_fft, 0],
                           inputs[i*self.num_fft:(i+1)*self.num_fft, 1])
            x = tf.signal.ifft(X)
            x_GI.append(tf.concat((x[self.num_fft - self.num_gard:self.num_fft], x), axis=0))
        ifft_signal = tf.concat(x_GI, axis=0)
        outputs = tf.concat((tf.math.real(ifft_signal)[:, None], tf.math.imag(ifft_signal)[:, None]), axis=1)
        return outputs

    def get_config(self):
        config = super(OFDMModulation, self).get_config()
        config.update({'num_gard':self.num_gard, 'num_fft': self.num_fft, 'num_sym':self.num_sym})
        return config


class OFDMDeModulation(layers.Layer):
    def __init__(self, ofdm_outshape, num_gard=16, num_fft=64, **kwargs):
        """保护间隔个数， fft个数， 每一帧ofdm符号个数等于num_gard+num_fft， 帧数等于符号个数除以num_fft"""
        super(OFDMDeModulation, self).__init__(**kwargs)
        self.num_gard, self.num_fft = num_gard, num_fft
        self.num_sym = num_gard + num_fft
        self.ofdm_outshape = ofdm_outshape

    def call(self, inputs):
        Y = []
        num_fram = self.ofdm_outshape // self.num_sym
        for i in range(num_fram):
            YGI = tf.complex(inputs[i*self.num_sym: (i+1)*self.num_sym, 0],
                           inputs[i*self.num_sym:(i+1)*self.num_sym, 1])
            y = tf.signal.fft(YGI[self.num_gard:self.num_sym])
            Y.append(y)
        fft_signal = tf.concat(Y, axis=0)
        outputs = tf.concat((tf.math.real(fft_signal)[:, None], tf.math.imag(fft_signal)[:, None]), axis=1)
        return outputs

    def get_config(self):
        config = super(OFDMDeModulation, self).get_config()
        config.update({'num_gard':self.num_gard, 'num_fft': self.num_fft,
                       'num_sym': self.num_sym,'ofdm_outshape':self.ofdm_outshape})
        return config


if __name__ == "__main__":
    mapping_pre = np.random.randn(128, 2)
    # papr_loss = PAPRConstraint()
    # loss = papr_loss(y_pred=mapping_pre, y_true=mapping_pre)
    # m = MappingLayer()
    # out = m(mapping_pre)
    # signal = np.random.randn(10, 2)
    # Noise = GaussianNoise(10, ofdm_model=False, num_syms=128)
    # out2 = Noise(signal)
    # PN = PowerNormalize()
    # out3 = PN(np.array([[1.,2.], [3.,4.], [5.,6.]]))
    ofdm = OFDMModulation(num_sym=128, num_gard=16, num_fft=64)
    ofdm_out = ofdm(mapping_pre)
    deofdm = OFDMDeModulation(ofdm_outshape=128//64*80, num_gard=16, num_fft=64)
    deofdm_out = deofdm(ofdm_out)
    print("finish!")