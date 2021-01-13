from my_code.tools import *
from my_code.model_tools import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)


"""
用于验证与数据的测试：
    1.生成随机数据
    2.载入训练好的编解码模型
    3.对encoder加载噪声，送入decoder解码计算误比特率并进行作图
"""
#  第一种验证：同一批数据，但是使用不同的噪声，验证模型对噪声的泛化能力
#  第二种验证：不同的数据，使用相同噪声，验证模型对数据的泛化能力


def validation_one():
    gh_path = GH_PATH(r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\G.mat',
                      r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\H.mat')
    fine_model = Model_save_path(r'../my_trained_model/mlp_ofdm_encoder_5000', r'../my_trained_model/mlp_ofdm_decoder_5000',
                                           r'../my_trained_model/mlp_ofdm_mapper_5000')
    model_load_path = Model_save_path(r'../my_model8/mlp_encoder', r'../my_model8/mlp_decoder',
                                      r'../my_model8/mlp_mapper')
    ofdm_model_load_path = Model_save_path(r'../my_model8/mlp_ofdm_encoder', r'../my_model8/mlp_ofdm_decoder',
                                           r'../my_model8/mlp_ofdm_mapper')
    ofdm_papr_model_load_path = Model_save_path(r'../my_model8/mlp_ofdm_papr_encoder',
                                                r'../my_model8/mlp_ofdm_papr_decoder',
                                                r'../my_model8/mlp_ofdm_papr_mapper')
    ofdm_conv_model_save_path = Model_save_path(r'../my_model_conv/ofdm_encoder', r'../my_model_conv/ofdm_decoder',
                                           r'../my_model_conv/ofdm_mapper')
    g = load_mat(gh_path.g_path)['outputG']
    m, bit_nums, channels = 3, 10000, 2  # 注意bit_nums应该与训练的时候大小一样
    ldpc_encoder = LDPCEncode(g, m, bit_nums)
    qam_padding_bits_test = ldpc_encoder.encode(model_name='mlp')
    num_syms = qam_padding_bits_test.shape[0]#*qam_padding_bits_test.shape[1]
    ofdm_model = 1
    encoder, decoder, mapper = model_load(ofdm_model_load_path)
    if ofdm_model:
        ofdm_ifft = OFDMModulation(num_syms, name='ofdm')
        ofdm_fft = OFDMDeModulation(num_syms // OFDMParameters.fft_num.value * OFDMParameters.ofdm_syms.value)
    history = {'snr': [], 'ber': [], 'papr': []}
    for snr in range(0, 25):
        channel = MyGaussianNoise(snr, ofdm_model=ofdm_model, num_sym=OFDMParameters.ofdm_syms.value,
                                  nbps=m, num_syms=num_syms)
        papr_esitimator = PAPRConstraint(num_syms, snr)
        mapping = encoder(qam_padding_bits_test)
        papr = papr_esitimator(mapping, mapping).numpy()
        if ofdm_model:
            mapping = ofdm_ifft(mapping)
        received = channel(mapping)
        if ofdm_model:
            received = ofdm_fft(received)
        recover_bits = decoder(received)
        qam_padding_bits_test_int = qam_padding_bits_test.astype('int')
        decision_bits = np.array(recover_bits.numpy() > 0.5).astype('int')
        ber = np.sum(np.sum(decision_bits != qam_padding_bits_test_int))/(qam_padding_bits_test.shape[0]*qam_padding_bits_test.shape[1])
        if snr % 5 == 0:
            plt.figure()
            scatter(received.numpy()[:, 0], received.numpy()[:, 1])
            plt.show()
        history['snr'].append(snr)
        history['ber'].append(ber)
        history['papr'].append(papr)
    plt.figure()
    plt.semilogy(history['snr'], history['ber'], 'k.-')
    plt.show()
    plt.figure()
    plt.plot(history['snr'], history['papr'], 'r--')
    plt.show()


if __name__ == "__main__":
    with tf.device('/CPU:0'):
        validation_one()
