from functools import reduce
from my_code.models import Models
from my_code.model_tools import *
from my_code.tools import *
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)

# np.random.seed(3)


class Train:
    def __init__(self, data: Data, lr, m, snr, opt="adam", continued=0, factor_trainable=1, train_union=0,
                 label_transform=False, mse=False, mapping_method='none', ofdm_model=0, reset_regular_factor=0):
        """
        损失函数用两种方法度量，binary_cross_entropy和mse以及其他基于llr的度量方法
        papr约束和mse使用自适应多任务学习方法
        """
        self.mse = mse
        self.data = data
        self.label_transform = label_transform
        self.factor_trainable = factor_trainable
        self.train_union = train_union
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
        self.mapping_loss = MappingConstraint(mapping_method)
        self.binary_cross_entropy_metrics = keras.metrics.BinaryCrossentropy()
        self.binary_accuracy_metrics = keras.metrics.BinaryAccuracy()
        if not continued:
            self.regularization_factor = [tf.Variable(1., trainable=factor_trainable),
                                          tf.Variable(1., trainable=factor_trainable)]
            if ofdm_model:
                self.regularization_factor.append(tf.Variable(1., trainable=factor_trainable))
        else:
            if 'regularization_factor.pkl' not in os.listdir("../data/class_element"):
                raise Exception("先保存模型！")
            self.regularization_factor = Saver.load_class_element("../data/class_element/regularization_factor.pkl")
            if factor_trainable:
                tmp = list(map(lambda x: tf.Variable(x.value(), trainable=factor_trainable), self.regularization_factor))
                self.regularization_factor = tmp
        if reset_regular_factor:
            self.regularization_factor = [tf.Variable(1., trainable=factor_trainable),
                                          tf.Variable(1., trainable=factor_trainable),
                                          tf.Variable(1., trainable=factor_trainable)]
        if ofdm_model:
            self.papr_loss = PAPRConstraint(num_syms=data[0].shape[0], snr=snr)

    def train_loop(self, epochs, encoder, decoder, mapper):
        history = History([], [], [], [], [], [], [])
        x_train, y_train, x_val, y_val = list(map(lambda x: tf.convert_to_tensor(x), self.data))
        if self.mse and self.label_transform:
            y_train = 16 * y_train - 8
            y_val = 16 * y_val - 8
        if not self.mse:
            y_train = tf.cast(y_train, tf.int32)
            y_val = tf.cast(y_val, tf.int32)
        for epoch in range(epochs):
            train_loss, train_accuracy, train_papr = self.__train_step(encoder, decoder, mapper, x_train, y_train)
            mapping = encoder(x_val)
            val_pre = decoder(mapper(x_val))
            #  val metrics
            val_loss = self.binary_cross_entropy_metrics(y_val, val_pre).numpy()
            val_accuracy = self.binary_accuracy_metrics(y_val, val_pre).numpy()
            val_papr = -1.
            if self.ofdm_model:
                val_papr = self.papr_loss(mapping, mapping).numpy()
            history.epoch.append(epoch)
            history.loss.append(float(train_loss))
            history.val_loss.append(float(val_loss))
            history.accuracy.append(float(train_accuracy))
            history.val_accuracy.append(float(val_accuracy))
            history.papr.append(float(train_papr))
            history.val_papr.append(float(val_papr))
            if epoch % 50 == 0:
                print("factors:{}".format(list(map(lambda x: x.numpy(), self.regularization_factor))))
                print("epoch:{}, loss:{}, val_loss:{},\n acc:{}, val_acc:{}, papr:{}, val_papr:{}".
                      format(epoch, train_loss, val_loss, train_accuracy, val_accuracy, train_papr, val_papr))
        plot(history, 0)

        return history

    def __train_step(self, encoder, decoder, mapper, x_train, y_train):
        with tf.GradientTape() as tape1:
            prediction = decoder(mapper(x_train))
            mapping = encoder(x_train)
            binary_loss = self.prime_loss(y_train, prediction)
            mapping_loss = self.mapping_loss(mapping, mapping)
            current_loss = 1 / (2 * self.regularization_factor[0] ** 2) * binary_loss + \
                           1 / (2 * self.regularization_factor[1] ** 2) * mapping_loss + \
                           tf.math.log(reduce(lambda x, y: x * y ** 2, self.regularization_factor))
            train_papr = -1.
            if self.ofdm_model and self.train_union:  # ofdm训练ber
                papr_loss = self.papr_loss(mapping, mapping)
                train_papr = papr_loss.numpy()
                current_loss = current_loss + 1 / (2 * self.regularization_factor[-1] ** 2) * papr_loss + \
                    tf.math.log(self.regularization_factor[-1]**2)
            train_loss = self.binary_cross_entropy_metrics(y_train, prediction).numpy()
            train_accuracy = self.binary_accuracy_metrics(y_train, prediction).numpy()
        trainable_variables = [mapper.trainable_variables, decoder.trainable_variables]
        if self.factor_trainable:
            trainable_variables.append(self.regularization_factor)
        model_gradients = tape1.gradient(current_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(model_gradients[0], mapper.trainable_variables))
        self.optimizer.apply_gradients(zip(model_gradients[1], decoder.trainable_variables))
        if self.factor_trainable:
            self.optimizer.apply_gradients(zip(model_gradients[2], self.regularization_factor))
        return train_loss, train_accuracy, train_papr


def main():
    gh_path = GH_PATH(r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\G.mat',
                      r'D:\LYJ\AutoEncoder-Based-Communication-System-master\matlab_code\genarateH G\H.mat')
    g = load_mat(gh_path.g_path)['outputG']
    h = load_mat(gh_path.h_path)['outputH']
    m, bit_nums, snr_ebn0, channels = 3, 10000, 10, 2  # 表示一帧数据
    result_save_path = Result_save_path(r'../result_data/%dsnr_encoder_train_mapping.mat' % 23,
                                        r'../result_data/%dsnr_encoder_test_mapping.mat' % 23,
                                        r'../result_data/%dsnr_decoder_train_recover.mat' % 23,
                                        r'../result_data/%dsnr_decoder_test_recover.mat' % 23,
                                        r'../result_data/train_bits.mat',
                                        r'../result_data/test_bits.mat')
    norm_model_save_path = Model_save_path(r'../my_model8/mlp_encoder', r'../my_model8/mlp_decoder',
                                      r'../my_model8/mlp_mapper')
    ofdm_model_save_path = Model_save_path(r'../my_model8/mlp_ofdm_encoder', r'../my_model8/mlp_ofdm_decoder',
                                           r'../my_model8/mlp_ofdm_mapper')
    ofdm_papr_model_save_path = Model_save_path(r'../my_model8/mlp_ofdm_papr_encoder', r'../my_model8/mlp_ofdm_papr_decoder',
                                           r'../my_model8/mlp_ofdm_papr_mapper')
    ofdm_conv_model_save_path = Model_save_path(r'../my_model_conv/ofdm_encoder', r'../my_model_conv/ofdm_decoder',
                                           r'../my_model_conv/ofdm_mapper')
    model_name = 'conv1d'  # conv1d or mlp
    ldpc_encoder = LDPCEncode(g, m, bit_nums)
    qam_padding_bits = ldpc_encoder.encode(model_name=model_name)  # 随机比特流数据作为训练数据
    qam_padding_bits_val = ldpc_encoder.encode(model_name=model_name)  # 生成验证数据
    qam_padding_bits_test = ldpc_encoder.encode(model_name=model_name)  # 测试数据
    print("training data shape: {}".format(qam_padding_bits.shape))
    original_bits = ldpc_encoder.bits
    data_set = Data(qam_padding_bits, qam_padding_bits, qam_padding_bits_val, qam_padding_bits_val)
    num_signals, k = qam_padding_bits.shape[0], int(qam_padding_bits.shape[1] // m)
    input_shape = qam_padding_bits_test.shape if model_name == 'conv1d' else num_signals
    continued, ofdm_model, train_union = 1, 0, 0  # ofdm 模式下需要先训练ber，再训练papr（0,1,0）-->（1,1,1）
    factor_trainable = 0
    constraint, mapping_method = 'pow', 'pow'
    model_load_path = ofdm_conv_model_save_path
    model_save_path = ofdm_conv_model_save_path
    if continued:
        encoder, decoder, mapper = model_load(model_path=model_load_path)
        if train_union:  # 冻结decoder
            for layer in decoder.layers[:2]:
                layer.trainable = False
    else:
        M = Models(m=m, constraint=constraint, channels=2)  # constraint:power or amplitude
        encoder, decoder, mapper = M.get_model(model_name=model_name, snr=snr_ebn0, ofdm_model=ofdm_model,
                                               input_shape=input_shape,
                                               ofdm_outshape=num_signals // OFDMParameters.fft_num.value * OFDMParameters.ofdm_syms.value)
    T = Train(data=data_set, lr=0.001, snr=snr_ebn0, m=m, train_union=train_union, factor_trainable=factor_trainable,
              mapping_method=mapping_method, continued=continued, ofdm_model=ofdm_model)  # papr_method:none\pow\papr
    history = T.train_loop(epochs=10001, encoder=encoder, decoder=decoder, mapper=mapper)
    if model_name == 'mlp':
        encoder = keras.Model(inputs=mapper.inputs,
                              outputs=mapper.get_layer(name='normalize').output if constraint is not 'none'
                              else mapper.get_layer(name='encoder_out').output, name='encoder')
    else:
        encoder = mapper.encoder
    Saver.save_model(encoder, decoder, mapper, data_set.train_data, qam_padding_bits_test,
                     model_save_path=model_save_path, result_save_path=result_save_path)
    Saver.save_result(result_save_path, qam_padding_bits, qam_padding_bits_test)
    Saver.save_class_element(T, "regularization_factor", "../data/class_element/regularization_factor.pkl")
    Saver.save_history("../data/history.json", history)


if __name__ == "__main__":
    # TODO：
    #  1.优化papr
    #  2.papr的自适应MTL表达式
    with tf.device('/CPU:0'):
        main()
