from re import A
from keras import backend as K
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Layer
from keras.layers import Reshape, GlobalMaxPool1D, Dense, add, Activation, multiply, Lambda, Conv1D, concatenate
import tensorflow as tf


# tf.config.experimental_run_functions_eagerly(True)

class GIIF_module(Layer):
    def __init__(self, **kwargs):
        super(GIIF_module, self).__init__(**kwargs)

    def build(self, input_shape):
        num_channels = input_shape[2]
        self.alpha = self.add_weight(shape=(1,),
                                     initializer='zeros',
                                     regularizer=None,
                                     constraint=None,
                                     name='alpha',
                                     trainable=True)
        self.gamma = self.add_weight(shape=(1,),
                                     initializer='zeros',
                                     regularizer=None,
                                     constraint=None,
                                     name='gamma',
                                     trainable=True)

        self.conv1 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv1D(filters=num_channels, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')

        self.conv4 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv1D(filters=num_channels, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')

        self.conv7 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv1D(filters=num_channels, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')

        self.conv10 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv11 = Conv1D(filters=num_channels // 8, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')
        self.conv12 = Conv1D(filters=num_channels, kernel_size=1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal')

        self.built = True
        super(GIIF_module, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

    def call(self, input):
        input_shape = input.get_shape().as_list()
        ECG_feature = input[:, :, :, 0]
        PCG_feature = input[:, :, :, 1]
        num_channels = ECG_feature.shape[-1]
        Q_ECG_intra = self.conv1(ECG_feature)
        K_ECG_intra = self.conv2(ECG_feature)
        V_ECG_intra = self.conv3(ECG_feature)
        K_T_ECG_intra = tf.transpose(K_ECG_intra, (0, 2, 1))  # shape(batch_size, num_channels//8, number_timestep)
        QK_ECG_intra = K.batch_dot(Q_ECG_intra, K_T_ECG_intra)  # shape(batch_size, number_timestep, number_timestep)
        D = num_channels // 8
        QK_ECG_intra = QK_ECG_intra / (D ** 0.5)
        weighted_QK_ECG_intra = Activation('softmax')(QK_ECG_intra)
        QKV_ECG_intra = K.batch_dot(weighted_QK_ECG_intra, V_ECG_intra)  # shape(batch_size, number_timestep, num_channels)

        Q_ECG_inter = self.conv4(ECG_feature)
        K_PCG_inter = self.conv5(PCG_feature)
        V_PCG_inter = self.conv6(PCG_feature)
        K_T_PCG_inter = tf.transpose(K_PCG_inter, (0, 2, 1))  # shape(batch_size, num_channels//8, number_timestep)
        QK_ECG_inter = K.batch_dot(Q_ECG_inter, K_T_PCG_inter)  # shape(batch_size, number_timestep, number_timestep)
        D = num_channels // 8
        QK_ECG_inter = QK_ECG_inter / (D ** 0.5)
        weighted_QK_ECG_inter = Activation('softmax')(QK_ECG_inter)
        QKV_ECG_inter = K.batch_dot(weighted_QK_ECG_inter, V_PCG_inter)  # shape(batch_size, number_timestep, num_channels)

        ECG_feature_output = QKV_ECG_inter + self.alpha * QKV_ECG_intra

        Q_PCG_intra = self.conv7(PCG_feature)
        K_PCG_intra = self.conv8(PCG_feature)
        V_PCG_intra = self.conv9(PCG_feature)
        K_T_PCG_intra = tf.transpose(K_PCG_intra, (0, 2, 1))  # shape(batch_size, num_channels//8, number_timestep)
        QK_PCG_intra = K.batch_dot(Q_PCG_intra, K_T_PCG_intra)  # shape(batch_size, number_timestep, number_timestep)
        D = num_channels // 8
        QK_PCG_intra = QK_PCG_intra / (D ** 0.5)
        weighted_QK_PCG_intra = Activation('softmax')(QK_PCG_intra)
        QKV_PCG_intra = K.batch_dot(weighted_QK_PCG_intra, V_PCG_intra)  # shape(batch_size, number_timestep, num_channels)

        Q_PCG_inter = self.conv10(PCG_feature)
        K_ECG_inter = self.conv11(ECG_feature)
        V_ECG_inter = self.conv12(ECG_feature)
        K_T_ECG_inter = tf.transpose(K_ECG_inter, (0, 2, 1))  # shape(batch_size, num_channels//8, number_timestep)
        QK_PCG_inter = K.batch_dot(Q_PCG_inter, K_T_ECG_inter)  # shape(batch_size, number_timestep, number_timestep)
        D = num_channels // 8
        QK_PCG_inter = QK_PCG_inter / (D ** 0.5)
        weighted_QK_PCG_inter = Activation('softmax')(QK_PCG_inter)
        QKV_PCG_inter = K.batch_dot(weighted_QK_PCG_inter, V_ECG_inter)  # shape(batch_size, number_timestep, num_channels)

        PCG_feature_output = QKV_PCG_inter + self.gamma * QKV_PCG_intra

        out = concatenate([ECG_feature_output, PCG_feature_output], axis=-1)
        return out


def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer


def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer


def slice_layer(x, group_number, channel_input):
    output_list = []
    single_channel = channel_input // group_number
    for j in range(group_number):
        out = x[:, :, j * single_channel:(j + 1) * single_channel]
        output_list.append(out)
    return output_list


def group_convolution_block(layer, **params):
    from keras.layers import concatenate, add
    subsample_length = 1
    num_filters = layer.shape[-1]
    group_number = params["group_number"]
    slice_list = slice_layer(layer, group_number, num_filters)  # 将输入特征进行分组
    side = add_conv_weight(slice_list[1],
                           params["conv_filter_length"],
                           num_filters // group_number,
                           subsample_length,
                           **params)
    side = _bn_relu(side, **params)
    z = concatenate([slice_list[0], side], axis=-1)  # for one and second stage
    for m in range(2, len(slice_list)):
        y = add_conv_weight(add([side, slice_list[m]]),
                            params["conv_filter_length"],
                            num_filters // group_number,
                            subsample_length,
                            **params)
        y = _bn_relu(y, **params)
        side = y
        z = concatenate([z, y], axis=-1)
    out = add_conv_weight(z,
                          params["conv_filter_length"],
                          num_filters,
                          subsample_length,
                          **params)
    return out


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            params["conv_num_filters_start"],
            subsample_length=subsample_length,
            **params)
        layer = _bn_relu(layer, **params)
    return layer


def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda
    from keras.layers import LSTM

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
               and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer


def res2net_block_v2(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda
    from keras.layers import LSTM

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = ((num_filters / layer.shape[-1]) == 2)  # 与backbone提取的特定模态特征通道数变化一致
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
    # 是否需要降采样和改变通道数量
    if subsample_length == 2:
        layer = MaxPooling1D(pool_size=subsample_length)(layer)
    if zero_pad is True:
        layer = _bn_relu(
            layer,
            dropout=0,
            **params)
        layer = add_conv_weight(
            layer,
            1,
            num_filters,
            1,
            **params)

    if not (block_index == 1):
        layer = _bn_relu(
            layer,
            dropout=params["conv_dropout"],
            **params)
    layer = group_convolution_block(
        layer,
        **params)
    # print('layer=', layer.shape)

    layer = Add()([shortcut, layer])
    return layer


def get_num_filters_at_index(index, num_start_filters, **params):
    return 2 ** int(index / params["conv_increase_channels_at"]) \
        * num_start_filters


def region_aware_two(ECG_previous_layer, PCG_previous_layer, **params):
    from keras.layers import GlobalAveragePooling1D, Reshape, GlobalMaxPool1D, Dense, add, Activation, multiply, Lambda, Conv1D, concatenate, Reshape
    ECG_PCG_layer_concat = concatenate([ECG_previous_layer, PCG_previous_layer], axis=-1)
    channel = ECG_PCG_layer_concat.shape[-1]
    weight_feature = Conv1D(filters=channel / 8, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(ECG_PCG_layer_concat)
    weight_feature = Activation(params["conv_activation"])(weight_feature)
    weight_feature = Conv1D(filters=channel / 8, kernel_size=16, strides=1, padding='same', kernel_initializer='he_normal')(weight_feature)
    weight_feature = Activation(params["conv_activation"])(weight_feature)
    weight_feature = Conv1D(filters=2, kernel_size=1, strides=1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(weight_feature)
    ECG_weight = Reshape((weight_feature.shape[1], 1))(weight_feature[:, :, 0])
    PCG_weight = Reshape((weight_feature.shape[1], 1))(weight_feature[:, :, 1])
    # print('ECG_weight=', ECG_weight.shape)
    ECG_layer = multiply([ECG_previous_layer, ECG_weight])
    PCG_layer = multiply([PCG_previous_layer, PCG_weight])

    # channel_attention
    ECG_channel_weight, PCG_channel_weight = channel_attention(ECG_previous_layer, PCG_previous_layer)
    ECG_layer = multiply([ECG_layer, ECG_channel_weight])
    PCG_layer = multiply([PCG_layer, PCG_channel_weight])
    return ECG_layer, PCG_layer


def channel_attention(ECG_previous_layer, PCG_previous_layer):
    from keras.layers import GlobalAveragePooling1D, Reshape, GlobalMaxPool1D, Dense, add, Activation, multiply, concatenate
    ECG_PCG_layer_concat = concatenate([ECG_previous_layer, PCG_previous_layer], axis=-1)
    channel = ECG_PCG_layer_concat.shape[-1]
    Dense_one = Dense(channel // 8, kernel_initializer='he_normal', activation='relu')
    Dense_two = Dense(channel, kernel_initializer='he_normal')

    max_pool = GlobalMaxPool1D()(ECG_PCG_layer_concat)
    max_pool = tf.expand_dims(max_pool, axis=1)
    assert max_pool.shape[1:] == (1, channel)
    max_pool = Dense_one(max_pool)
    assert max_pool.shape[1:] == (1, channel // 8)
    max_pool = Dense_two(max_pool)
    assert max_pool.shape[1:] == (1, channel)

    avg_pool = GlobalAveragePooling1D()(ECG_PCG_layer_concat)
    avg_pool = tf.expand_dims(avg_pool, axis=1)
    assert avg_pool.shape[1:] == (1, channel)
    avg_pool = Dense_one(avg_pool)
    assert avg_pool.shape[1:] == (1, channel // 8)
    avg_pool = Dense_two(avg_pool)
    assert avg_pool.shape[1:] == (1, channel)

    cbam_feature = add([max_pool, avg_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    # print('cbam_feature=',cbam_feature.shape)
    # print('layer=',layer.shape)
    ECG_layer_weight = cbam_feature[:, :, 0:ECG_previous_layer.shape[-1]]
    PCG_layer_weight = cbam_feature[:, :, PCG_previous_layer.shape[-1]:]
    return ECG_layer_weight, PCG_layer_weight


def stem_structure(layer, **params):
    from keras.layers import LSTM, concatenate, Bidirectional
    layer_conv = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer_conv = _bn_relu(layer_conv, **params)
    layer_conv = add_conv_weight(
        layer_conv,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer_conv = _bn_relu(layer_conv, **params)

    layer_LSTM = LSTM(units=params["conv_num_filters_start"], return_sequences=True)(layer)
    # layer_LSTM = LSTM(units=params["conv_num_filters_start"],return_sequences=True)(layer_LSTM)

    layer = concatenate([layer_conv, layer_LSTM], axis=-1)

    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    return layer


def LDA(index, Aggregation_feature, Feature_list, **params):
    from keras.layers import Add, Activation, multiply, UpSampling1D
    weight_map = Feature_list[index + 2]
    if Aggregation_feature.shape[1] != weight_map.shape[1]:
        upsampling_size = (Aggregation_feature.shape[1]) // (weight_map.shape[1])
        weight_map = UpSampling1D(size=upsampling_size)(weight_map)  # 上采样以合并不同level特征
    weight_map = add_conv_weight(weight_map, 1, 1, subsample_length=1, **params)
    weight_map = Activation('softmax')(weight_map)
    weighted_aggregation_feature = multiply([Aggregation_feature, weight_map])
    Aggregation_feature = Add()([Aggregation_feature, weighted_aggregation_feature])
    return Aggregation_feature


def dense_fusion_model(ECG_feature_list, PCG_feature_list, **params):
    import numpy as np
    from keras.layers import LSTM, concatenate, Bidirectional, MaxPooling1D, Add
    ECG_previous_layer = ECG_feature_list[0]
    PCG_previous_layer = PCG_feature_list[0]
    # ECG_previous_layer = LDA(-1,ECG_previous_layer,ECG_feature_list,**params)
    # PCG_previous_layer = LDA(-1,PCG_previous_layer,PCG_feature_list,**params)
    ECG_layer_weighted, PCG_layer_weighted = region_aware_two(ECG_previous_layer, PCG_previous_layer, **params)  # 加权后的
    ECG_PCG_layer = concatenate([ECG_layer_weighted, PCG_layer_weighted], axis=-1)
    # ECG_PCG_layer = concatenate([ECG_layer, PCG_layer], axis=-1)
    ECG_PCG_layer = add_conv_weight(
        ECG_PCG_layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    ECG_PCG_layer = _bn_relu(ECG_PCG_layer, **params)
    num_filters_init = params["conv_num_filters_start"]

    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        # 聚合不同尺度特征
        if subsample_length == 2:
            # 调节前一级聚合特征尺寸与当前级别特定模态特征一致
            ECG_previous_layer = MaxPooling1D(pool_size=2)(ECG_previous_layer)  # 下采样以合并不同level特征
            PCG_previous_layer = MaxPooling1D(pool_size=2)(PCG_previous_layer)  # 下采样以合并不同level特征
            # 调节前一级聚合特征通道数与当前级别特定模态特征一致
            if num_filters != num_filters_init:
                ECG_previous_layer = add_conv_weight(ECG_previous_layer, 1, num_filters, subsample_length=1, **params)
                PCG_previous_layer = add_conv_weight(PCG_previous_layer, 1, num_filters, subsample_length=1, **params)
                num_filters_init = num_filters
            ECG_previous_layer = Add()([ECG_previous_layer, ECG_feature_list[index + 1]])
            PCG_previous_layer = Add()([PCG_previous_layer, PCG_feature_list[index + 1]])
            # 局部信息引导
            # if index != 15:
            #     ECG_previous_layer = LDA(index,ECG_previous_layer,ECG_feature_list,**params)
            #     PCG_previous_layer = LDA(index,PCG_previous_layer,PCG_feature_list,**params)
            # 跨模态区域感知
            if index <= 7:
                # 优化模型多尺度特征提取能力
                ECG_PCG_layer = res2net_block_v2(
                    ECG_PCG_layer,
                    num_filters,
                    subsample_length,
                    index,
                    **params)
                ECG_layer_weighted, PCG_layer_weighted = region_aware_two(ECG_previous_layer, PCG_previous_layer, **params)  # 加权后的
                ECG_PCG_layer = concatenate([ECG_PCG_layer, ECG_layer_weighted, PCG_layer_weighted], axis=-1)
                ECG_PCG_layer = _bn_relu(ECG_PCG_layer, **params)
                ECG_PCG_layer = add_conv_weight(
                    ECG_PCG_layer,
                    params["conv_filter_length"],
                    num_filters,
                    subsample_length=1,
                    **params)
            if index > 7:
                if index == 9:
                    ECG_PCG_layer = res2net_block_v2(
                        ECG_PCG_layer,
                        num_filters,
                        subsample_length,
                        index,
                        **params)
                    channel_share_input = ECG_PCG_layer
                else:
                    ECG_PCG_layer = MaxPooling1D(pool_size=2)(ECG_PCG_layer)
                    if num_filters != ECG_PCG_layer.shape[-1]:
                        ECG_PCG_layer = add_conv_weight(ECG_PCG_layer, 1, num_filters, subsample_length=1, **params)
                ECG_PCG_previous_layer = concatenate([ECG_previous_layer[..., np.newaxis], PCG_previous_layer[..., np.newaxis]], axis=-1)  # 按照通道拼接特征
                ECG_PCG_layer_weighted = GIIF_module()(ECG_PCG_previous_layer)  # 加权后的
                ECG_PCG_layer = concatenate([ECG_PCG_layer, ECG_PCG_layer_weighted], axis=-1)
                ECG_PCG_layer = _bn_relu(ECG_PCG_layer, **params)
                ECG_PCG_layer = add_conv_weight(
                    ECG_PCG_layer,
                    1,
                    num_filters,
                    subsample_length=1,
                    **params)

    # ECG_layer = _bn_relu(ECG_layer, **params)
    # PCG_layer = _bn_relu(PCG_layer, **params)
    # ECG_PCG_layer= _bn_relu(ECG_PCG_layer, **params)
    return ECG_PCG_layer, channel_share_input


def channel_share_model(channel_share_input, **params):
    # 通道融合之前经过SE注意力机制
    layer = _bn_relu(channel_share_input, **params)
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        128,
        subsample_length=1,
        **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        if index >= 10:
            num_filters = get_num_filters_at_index(
                index, params["conv_num_filters_start"], **params)
            layer = resnet_block(
                layer,
                num_filters,
                subsample_length,
                index,
                **params)
    return layer


def add_output_layer(ECG_PCG_layer, final_layer_name=None, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import LSTM

    # ECG_PCG_layer = LSTM(units=64,return_sequences=True)(ECG_PCG_layer)
    # ECG_PCG_layer = LSTM(units=32,return_sequences=False)(ECG_PCG_layer)
    # ECG_PCG_layer = Dense(params["num_categories"])(ECG_PCG_layer)
    # ECG_PCG_layer = Activation('softmax')(ECG_PCG_layer)
    ECG_PCG_layer = GlobalMaxPool1D()(ECG_PCG_layer)
    ECG_PCG_layer = Dense(128)(ECG_PCG_layer)
    ECG_PCG_layer = Activation(params["conv_activation"])(ECG_PCG_layer)
    ECG_PCG_layer = Dense(32)(ECG_PCG_layer)
    ECG_PCG_layer = Activation(params["conv_activation"])(ECG_PCG_layer)
    ECG_PCG_layer = Dense(params["num_categories"])(ECG_PCG_layer)
    ECG_PCG_layer = Activation('softmax', name=final_layer_name)(ECG_PCG_layer)
    '''
    ECG_layer = LSTM(units=64,return_sequences=True)(ECG_layer)
    ECG_layer = LSTM(units=32,return_sequences=False)(ECG_layer)
    ECG_layer = Dense(params["num_categories"])(ECG_layer)
    ECG_layer = Activation('softmax')(ECG_layer)

    PCG_layer = LSTM(units=64,return_sequences=True)(PCG_layer)
    PCG_layer = LSTM(units=32,return_sequences=False)(PCG_layer)
    PCG_layer = Dense(params["num_categories"])(PCG_layer)
    PCG_layer = Activation('softmax')(PCG_layer)
    '''
    # layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    # layer = GlobalAveragePooling1D()(layer)
    return ECG_PCG_layer


def add_compile(model, **params):
    from keras.optimizers import Adam
    # from models.focal_losses import binary_focal_loss
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # model.compile(loss=[binary_focal_loss(gamma=2, alpha=0.25)],
    #              optimizer=optimizer, 
    #              metrics=['accuracy'],
    #              loss_weights = [0.5,0.5,0.5])


def LGC2_Net(**params):
    from keras.models import Model
    from keras.layers import Input, concatenate
    import numpy as np
    # 搭建ECG特定模态编码器
    inp_ECG = Input(shape=(2560, 1), name='ECG_backbone_input')
    ECG_feature_list = []
    layer_ECG = stem_structure(inp_ECG, **params)
    ECG_feature_list.append(layer_ECG)  # 时空特征
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer_ECG = resnet_block(
            layer_ECG,
            num_filters,
            subsample_length,
            index,
            **params)
        ECG_feature_list.append(layer_ECG)
    backbone_model_ECG = Model(inputs=inp_ECG, outputs=ECG_feature_list)

    # 搭建PCG特定模态编码器
    inp_PCG = Input(shape=(2560, 1), name='PCG_backbone_input')
    PCG_feature_list = []
    layer_PCG = stem_structure(inp_PCG, **params)
    PCG_feature_list.append(layer_PCG)  # 第一层时空特征
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer_PCG = resnet_block(
            layer_PCG,
            num_filters,
            subsample_length,
            index,
            **params)
        PCG_feature_list.append(layer_PCG)  # 卷积块数量+1层
    backbone_model_PCG = Model(inputs=inp_PCG, outputs=PCG_feature_list)

    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    print('input=', inputs.shape)
    ECG_feature_APEX = backbone_model_ECG(inputs[:, :, 0, 0:1])
    PCG_feature_APEX = backbone_model_PCG(inputs[:, :, 0, 1:2])
    ECG_PCG_fusion_APEX, channel_share_APEX = dense_fusion_model(ECG_feature_APEX, PCG_feature_APEX, **params)
    ECG_feature_LLSB = backbone_model_ECG(inputs[:, :, 1, 0:1])
    PCG_feature_LLSB = backbone_model_PCG(inputs[:, :, 1, 1:2])
    ECG_PCG_fusion_LLSB, channel_share_LLSB = dense_fusion_model(ECG_feature_LLSB, PCG_feature_LLSB, **params)
    ECG_feature_LUSB = backbone_model_ECG(inputs[:, :, 2, 0:1])
    PCG_feature_LUSB = backbone_model_PCG(inputs[:, :, 2, 1:2])
    ECG_PCG_fusion_LUSB, channel_share_LUSB = dense_fusion_model(ECG_feature_LUSB, PCG_feature_LUSB, **params)
    ECG_feature_RUSB = backbone_model_ECG(inputs[:, :, 3, 0:1])
    PCG_feature_RUSB = backbone_model_PCG(inputs[:, :, 3, 1:2])
    ECG_PCG_fusion_RUSB, channel_share_RUSB = dense_fusion_model(ECG_feature_RUSB, PCG_feature_RUSB, **params)

    # 通道共享空间
    channel_share_input = concatenate([channel_share_APEX, channel_share_LLSB, channel_share_LUSB, channel_share_RUSB], axis=-1)
    channel_share_feature = channel_share_model(channel_share_input, **params)

    ECG_PCG_layer_fusion = concatenate([ECG_PCG_fusion_APEX, ECG_PCG_fusion_LLSB, ECG_PCG_fusion_LUSB, ECG_PCG_fusion_RUSB, channel_share_feature], axis=-1)
    ECG_PCG_layer_fusion = _bn_relu(ECG_PCG_layer_fusion, **params)
    ECG_PCG_layer_fusion = add_conv_weight(
        ECG_PCG_layer_fusion,
        1,
        256,
        subsample_length=1,
        **params)
    ECG_PCG_fusion_APEX = _bn_relu(ECG_PCG_fusion_APEX, **params)
    ECG_PCG_fusion_LLSB = _bn_relu(ECG_PCG_fusion_LLSB, **params)
    ECG_PCG_fusion_LUSB = _bn_relu(ECG_PCG_fusion_LUSB, **params)
    ECG_PCG_fusion_RUSB = _bn_relu(ECG_PCG_fusion_RUSB, **params)
    channel_share_feature = _bn_relu(channel_share_feature, **params)
    ECG_PCG_layer_fusion = _bn_relu(ECG_PCG_layer_fusion, **params)
    # 各分支输出
    ECG_PCG_layer_fusion_output = add_output_layer(ECG_PCG_layer_fusion, "layer_fusion_output", **params)
    ECG_PCG_fusion_APEX_output = add_output_layer(ECG_PCG_fusion_APEX, "fusion_APEX_output", **params)
    ECG_PCG_fusion_LLSB_output = add_output_layer(ECG_PCG_fusion_LLSB, "fusion_LLSB_output", **params)
    ECG_PCG_fusion_LUSB_output = add_output_layer(ECG_PCG_fusion_LUSB, "fusion_LUSB_output", **params)
    ECG_PCG_fusion_RUSB_output = add_output_layer(ECG_PCG_fusion_RUSB, "fusion_RUSB_output", **params)
    ECG_PCG_channel_share_output = add_output_layer(channel_share_feature, "channel_share_output", **params)
    model = Model(inputs=[inputs],
                  outputs=[ECG_PCG_layer_fusion_output, ECG_PCG_fusion_APEX_output, ECG_PCG_fusion_LLSB_output, ECG_PCG_fusion_LUSB_output, ECG_PCG_fusion_RUSB_output, ECG_PCG_channel_share_output])

    if params.get("compile", True):
        add_compile(model, **params)

    model.summary()
    return model
