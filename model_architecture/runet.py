import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.layers import Activation
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Reshape
from keras.models import Model
from keras import regularizers
from keras.layers import add
from keras.layers import Layer

def bn_relu():
    def bn_relu_func(x):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    return bn_relu_func

def res_conv(nb_filter, nb_row, nb_col, reg_weight, stride=(1, 1)):
    def _res_func(x):
        identity = x
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weight))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weight))(a)
        y = BatchNormalization()(a)

        return add([identity, y])
    return _res_func

def dconv_bn_nolinear(nb_filter, nb_row, nb_col, reg_weight, stride=(2, 2), activation="relu"):
    def _dconv_bn(x):
        x = UpSampling2D(size=stride)(x)
        x = ReflectionPadding2D(padding=(int(nb_row/2), int(nb_col/2)))(x)
        x = Conv2D(nb_filter, (nb_row, nb_col), padding='valid', kernel_regularizer=regularizers.l2(reg_weight))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x
    return _dconv_bn

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def build(self, input_shape):
        pass  # No trainable weights, no need to build

    def call(self, x):
        paddings = tf.constant([[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]])
        return tf.pad(x, paddings, mode='REFLECT')

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0] if input_shape[1] is not None else None,
            input_shape[2] + 2 * self.padding[1] if input_shape[2] is not None else None,
            input_shape[3]
        )

    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config


def create_vae(reg_weight, input_size=(128, 128, 2), output_channels=1):
    # Encoder
    inputs = Input(shape = input_size, name = 'image')

    enc1_conv = Conv2D(16, (3, 3), strides = (2,2), padding='same', kernel_regularizer=regularizers.l2(reg_weight))(inputs)
    enc1_bn_relu = bn_relu()(enc1_conv)

    enc2_conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(reg_weight))(enc1_bn_relu)
    enc2_bn_relu = bn_relu()(enc2_conv)

    enc3_conv = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(reg_weight))(enc2_bn_relu)
    enc3_bn_relu = bn_relu()(enc3_conv)

    enc4_conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(reg_weight))(enc3_bn_relu)
    enc4_bn_relu = bn_relu()(enc4_conv)

    enc5_conv = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(reg_weight))(enc4_bn_relu)
    enc5_bn_relu = bn_relu()(enc5_conv)

    enc6_conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(reg_weight))(enc5_bn_relu)
    enc6_bn_relu = bn_relu()(enc6_conv)

    x0 = res_conv(128, 3, 3, reg_weight)(enc6_bn_relu)
    x1 = res_conv(128, 3, 3, reg_weight)(x0)
    x2 = res_conv(128, 3, 3, reg_weight)(x1)
    x3 = res_conv(128, 3, 3, reg_weight)(x2)
    x4 = res_conv(128, 3, 3, reg_weight)(x3)
    x5 = res_conv(128, 3, 3, reg_weight)(x4)
    x6 = res_conv(128, 3, 3, reg_weight)(x5)

    merge6 = concatenate([enc6_bn_relu, x6], axis=3)
    dec6 = dconv_bn_nolinear(128, 3, 3, reg_weight, stride=(1, 1))(merge6)
    merge5 = concatenate([enc5_bn_relu, dec6], axis=3)
    dec5 = dconv_bn_nolinear(128, 3, 3, reg_weight, stride=(2, 2))(merge5)
    merge4 = concatenate([enc4_bn_relu, dec5], axis=3)
    dec4 = dconv_bn_nolinear(64, 3, 3, reg_weight, stride=(1, 1))(merge4)
    merge3 = concatenate([enc3_bn_relu, dec4], axis=3)
    dec3 = dconv_bn_nolinear(64, 3, 3, reg_weight, stride=(2, 2))(merge3)
    merge2 = concatenate([enc2_bn_relu, dec3], axis=3)
    dec2 = dconv_bn_nolinear(32, 3, 3, reg_weight, stride=(1, 1))(merge2)
    merge1 = concatenate([enc1_bn_relu, dec2], axis=3)
    dec1 = dconv_bn_nolinear(16, 3, 3, reg_weight, stride=(2, 2))(merge1)

    outputs = [Conv2D(output_channels, (3, 3), padding='same', activation='sigmoid')(dec1)]

    # Full net
    vae_model = Model(inputs=inputs, outputs=outputs)
    return vae_model
