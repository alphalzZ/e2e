from functools import reduce
from my_code.models import Models
from my_code.model_tools import *
from my_code.tools import *


# np.random.seed(3)


class Train:
    def __init__(self, data: Data, lr, m, snr, opt="adam", continued=0,
                 label_transform=True, mse=False, papr_method='none', ofdm_model=0):
        """
        损失函数用两种方法度量，binary_cross_entropy和mse以及其他基于llr的度量方法
        papr约束和mse使用自适应多任务学习方法
        """
        self.mse = mse
        self.data = data
        self.label_transform = label_transform
        self.__m = m
        self.ofdm_model = ofdm_model
        if opt == "adam":
            self.optimizer = keras.optimizers.Adam(lr=lr)
        elif opt == "sgd":
            self.optimizer = keras.optimizers.SGD(lr=lr)
        else:
            self.optimizer = keras.optimizers.Ftrl(lr=lr)
        self.prime_loss = keras.losses.MeanSquaredError() if mse \
            else keras.losses.BinaryCrossentropy()
        self.mapping_loss = MappingConstraint(papr_method=papr_method)
        self.binary_cross_entropy_metrics = keras.metrics.BinaryCrossentropy()
        self.binary_accuracy_metrics = keras.metrics.BinaryAccuracy()
        if not continued:
            self.regularization_factor = [tf.Variable(1., trainable=True),
                                          tf.Variable(1., trainable=True)]
        else:
            if 'regularization_factor.pkl' not in os.listdir("./data/class_element"):
                raise Exception("先保存模型！")
            self.regularization_factor = Saver.load_class_element("./data/class_element/regularization_factor.pkl")
        if ofdm_model:
            self.papr_loss = PAPRConstraint(num_syms=data[0].shape[0], snr=snr)
            self.regularization_factor.append(tf.Variable(1., trainable=True))

    def train_loop(self, epochs, encoder, decoder, mapper):
        history = History([], [], [], [], [])
        x_train, y_train, x_val, y_val = self.data
        if self.mse and self.label_transform:
            y_train = 16 * y_train - 8
            y_val = 16 * y_val - 8
        if not self.mse:
            y_train = y_train.astype('int')
            y_val = y_val.astype('int')
        for epoch in range(epochs):
            train_loss, train_accuracy = self.__train_step(encoder, decoder, mapper, x_train, y_train)
            val_pre = decoder(mapper(x_val))
            val_loss = self.binary_cross_entropy_metrics(val_pre, y_val)
            val_accuracy = self.binary_accuracy_metrics(val_pre, y_val)
            history.epoch.append(epoch)
            history.loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.accuracy.append(np.mean(train_accuracy))
            history.accuracy.append(np.mean(val_accuracy))
            if epoch % 50 == 0:
                print("epoch:{}, loss:{}, val_loss:{}, acc:{}, val_acc:{}".
                      format(epoch, train_loss, val_loss, train_accuracy, val_accuracy))
        plot(history, 0)

    def __train_step(self, encoder, decoder, mapper, x_train, y_train):
        with tf.GradientTape() as tape1:
            prediction = decoder(mapper(x_train))
            mapping = encoder(x_train)
            binary_loss = self.prime_loss(y_train, prediction)
            mapping_loss = self.mapping_loss(mapping, mapping)
            current_loss = 1 / (2 * self.regularization_factor[0] ** 2) * binary_loss + \
                           1 / (2 * self.regularization_factor[1] ** 2) * mapping_loss + \
                           tf.math.log(reduce(lambda x, y: x * y ** 2, self.regularization_factor))
            #  self.regularization_factor[0]**2*self.regularization_factor[1]**2*self.regularization_factor[2]**2
            if self.ofdm_model:
                papr_loss = self.papr_loss(mapping, mapping)
                current_loss = current_loss + 1 / (2 * self.regularization_factor[-1] ** 2) * papr_loss + \
                    tf.math.log(self.regularization_factor[-1]**2)
            train_loss = self.binary_cross_entropy_metrics(y_train, prediction)
            train_accuracy = self.binary_accuracy_metrics(y_train, prediction)
        model_gradients = tape1.gradient(current_loss, [mapper.trainable_variables,
                                                        decoder.trainable_variables,
                                                        self.regularization_factor])
        self.optimizer.apply_gradients(zip(model_gradients[0], mapper.trainable_variables))
        self.optimizer.apply_gradients(zip(model_gradients[1], decoder.trainable_variables))
        self.optimizer.apply_gradients(zip(model_gradients[2], self.regularization_factor))
        return train_loss, train_accuracy


def main():
    gh_path = GH_PATH(r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\G.mat',
                      r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\H.mat')
    g = load_mat(gh_path.g_path)['outputG']
    h = load_mat(gh_path.h_path)['outputH']
    m, bit_nums, snr, channels = 4, 10000, 23, 2
    result_save_path = Result_save_path(r'../result_data/%dsnr_encoder_train_mapping.mat' % 23,
                                        r'../result_data/%dsnr_encoder_test_mapping.mat' % 23,
                                        r'../result_data/%dsnr_decoder_train_recover.mat' % 23,
                                        r'../result_data/%dsnr_decoder_test_recover.mat' % 23,
                                        r'../result_data/train_bits.mat',
                                        r'../result_data/test_bits.mat')
    model_save_path = Model_save_path(r'../my_model/mlp_encoder', r'../my_model/mlp_decoder',
                                      r'../my_model/mlp_mapper')
    ofdm_model_save_path = Model_save_path(r'../my_model/mlp_ofdm_encoder', r'../my_model/mlp_ofdm_decoder',
                                           r'../my_model/mlp_ofdm_mapper')

    ldpc_encoder = LDPCEncode(g, m, bit_nums)
    qam_padding_bits = ldpc_encoder.encode(model_name='mlp')  # 随机比特流数据作为训练数据
    qam_padding_bits_val = ldpc_encoder.encode(model_name='mlp')  # 生成验证数据
    qam_padding_bits_test = ldpc_encoder.encode(model_name='mlp')  # 测试数据
    print("training data shape: {}".format(qam_padding_bits.shape))
    original_bits = ldpc_encoder.bits
    data_set = Data(qam_padding_bits, qam_padding_bits, qam_padding_bits_val, qam_padding_bits_val)
    num_signals, k = qam_padding_bits.shape[0], int(qam_padding_bits.shape[1] // m)
    continued, ofdm_model = 0, 1
    model_path = ofdm_model_save_path if ofdm_model else model_save_path
    if continued:
        encoder, decoder, mapper = model_load(model_path=model_path)
    else:
        M = Models(m=m, k=k, constraint='pow', channels=2)  # constraint:power or amplitude or other mapping methods
        encoder, decoder, mapper = M.get_model(model_name='mlp', snr=snr, ofdm_model=ofdm_model,
                                               input_shape=num_signals,
                                               ofdm_outshape=num_signals // OFDMParameters.fft_num.value * OFDMParameters.ofdm_syms.value)
    T = Train(data=data_set, lr=0.001, snr=snr, m=m, label_transform=False,
              mse=False, papr_method='pow', continued=continued)  # papr_method:none\pow\papr
    T.train_loop(epochs=51, encoder=encoder, decoder=decoder, mapper=mapper)

    Saver.save_model(encoder, decoder, mapper, data_set.train_data, qam_padding_bits_test,
                     model_save_path=model_path, result_save_path=result_save_path)
    Saver.save_result(result_save_path, qam_padding_bits, qam_padding_bits_test)
    Saver.save_class_element(T, "regularization_factor", "../data/class_element/regularization_factor.pkl")


if __name__ == "__main__":
    main()
