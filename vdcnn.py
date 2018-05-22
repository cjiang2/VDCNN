import keras
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Dense, Flatten
from keras.engine.topology import get_source_inputs
from k_maxpooling import *

def identity_block(inputs, filters, kernel_size=3, use_bias=False, shortcut=False):
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    relu = Activation('relu')(bn1)
    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(relu)
    out = BatchNormalization()(conv2)
    if shortcut:
        out = Add()([out, inputs])
    return Activation('relu')(out)

def conv_block(inputs, filters, kernel_size=3, use_bias=False, shortcut=False, 
               pool_type='max', sorted=True, stage=1):
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)

    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(relu1)
    out = BatchNormalization()(conv2)

    if shortcut:
        residual = Conv1D(filters=filters, kernel_size=1, strides=2, name='shortcut_conv1d_%d' % stage)(inputs)
        residual = BatchNormalization(name='shortcut_batch_normalization_%d' % stage)(residual)
        out = downsample(out, pool_type=pool_type, sorted=sorted, stage=stage)
        out = Add()([out, residual])
        out = Activation('relu')(out)
    else:
        out = Activation('relu')(out)
        out = downsample(out, pool_type=pool_type, sorted=sorted, stage=stage)
    if pool_type is not None:
        out = Conv1D(filters=2*filters, kernel_size=1, strides=1, padding='same', name='1_1_conv_%d' % stage)(out)
        out = BatchNormalization(name='1_1_batch_normalization_%d' % stage)(out)
    return out

def downsample(inputs, pool_type='max', sorted=True, stage=1):
    if pool_type == 'max':
        out = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool_%d' % stage)(inputs)
    elif pool_type == 'k_max':
        k = int(inputs._keras_shape[1]/2)
        out = KMaxPooling(k=k, sorted=sorted, name='pool_%d' % stage)(inputs)
    elif pool_type == 'conv':
        out = Conv1D(filters=inputs._keras_shape[-1], kernel_size=3, strides=2, padding='same', name='pool_%d' % stage)(inputs)
        out = BatchNormalization()(out)
    elif pool_type is None:
        out = inputs
    else:
        raise ValueError('unsupported pooling type!')
    return out

def VDCNN(num_classes, depth=9, sequence_length=1024, embedding_dim=16, 
          shortcut=False, pool_type='max', sorted=True, use_bias=False, input_tensor=None):
    if depth == 9:
        num_conv_blocks = (1, 1, 1, 1)
    elif depth == 17:
        num_conv_blocks = (2, 2, 2, 2)
    elif depth == 29:
        num_conv_blocks = (5, 5, 2, 2)
    elif depth == 49:
        num_conv_blocks = (8, 8, 5, 3)
    else:
        raise ValueError('unsupported depth for VDCNN.')

    inputs = Input(shape=(sequence_length, ), name='inputs')
    embedded_chars = Embedding(input_dim=sequence_length, output_dim=embedding_dim)(inputs)
    out = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name='temp_conv')(embedded_chars)

    # Convolutional Block 64
    for _ in range(num_conv_blocks[0] - 1):
        out = identity_block(out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut, 
                     pool_type=pool_type, sorted=sorted, stage=1)

    # Convolutional Block 128
    for _ in range(num_conv_blocks[1] - 1):
        out = identity_block(out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut, 
                     pool_type=pool_type, sorted=sorted, stage=2)

    # Convolutional Block 256
    for _ in range(num_conv_blocks[2] - 1):
        out = identity_block(out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut, 
                     pool_type=pool_type, sorted=sorted, stage=3)

    # Convolutional Block 512
    for _ in range(num_conv_blocks[3] - 1):
        out = identity_block(out, filters=512, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
    out = conv_block(out, filters=512, kernel_size=3, use_bias=use_bias, shortcut=False, 
                     pool_type=None, stage=4)

    # k-max pooling with k = 8
    out = KMaxPooling(k=8, sorted=True)(out)
    out = Flatten()(out)

    # Dense Layers
    out = Dense(2048, activation='relu')(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(num_classes, activation='softmax')(out)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = inputs

    # Create model.
    model = Model(inputs=inputs, outputs=out, name='VDCNN')
    return model

if __name__ == "__main__":
    model = VDCNN(10, depth=9, shortcut=False, pool_type='max')
    model.summary()