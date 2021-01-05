from tools import *
from model_tools import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter


"""
用于验证与数据的测试：
    1.生成随机数据
    2.载入训练好的编解码模型
    3.对encoder加载噪声，送入decoder解码计算误比特率并进行作图
"""
if __name__ == "__main__":
    gh_path = GH_PATH(r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\G.mat',
                      r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\H.mat')
    model_load_path = Model_save_path(r'./my_model/mlp_encoder', r'./my_model/mlp_decoder',
                                      r'./my_model/mlp_mapping_model')
    ofdm_model_load_path = Model_save_path(r'./my_model/mlp_ofdm_encoder', r'./my_model/mlp_ofdm_decoder',
                                           r'./my_model/mlp_ofdm_mapping_model')
    g = load_mat(gh_path.g_path)['outputG']
    m, bit_nums, channels = 4, 10000, 2  # 注意bit_nums应该与训练的时候大小一样
    ldpc_encoder = LDPCEncode(g, m, bit_nums)
    qam_padding_bits_test = ldpc_encoder.encode(model_name='mlp')
    num_syms = qam_padding_bits_test.shape[0]
    ofdm_model = 0
    mapping_model, encoder, decoder = model_load(ofdm_model_load_path if ofdm_model else model_load_path)
    if ofdm_model:
        ofdm_fft = OFDMDeModulation(num_syms // OFDMParameters.fft_num.value * OFDMParameters.ofdm_syms.value)
    history = {'snr': [], 'ber': []}
    for snr in range(0, 31, 2):
        channel = GaussianNoise(snr, ofdm_model=ofdm_model, num_sym=80, nbps=m, num_syms=num_syms)
        mapping = mapping_model(qam_padding_bits_test)
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
    plt.figure()
    plt.semilogy(history['snr'], history['ber'], 'ko')
    plt.show()



