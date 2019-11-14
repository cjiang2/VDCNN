import tensorflow as tf

from k_maxpooling import KMaxPooling


def identity_block(inputs, filters, kernel_size=3, use_bias=False, shortcut=False):
    conv1 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu = tf.keras.activations.relu(bn1)
    conv2 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(relu)
    out = tf.keras.layers.BatchNormalization()(conv2)
    if shortcut:
        out = tf.keras.layers.Add()([out, inputs])
    return tf.keras.activations.relu(out)


def conv_block(
    inputs,
    filters,
    kernel_size=3,
    use_bias=False,
    shortcut=False,
    pool_type="max",
    sort=True,
    stage=1,
):
    conv1 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.activations.relu(bn1)

    conv2 = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=1, padding="same"
    )(relu1)
    out = tf.keras.layers.BatchNormalization()(conv2)

    if shortcut:
        residual = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, strides=2, name="shortcut_conv1d_%d" % stage
        )(inputs)
        residual = tf.keras.layers.BatchNormalization(
            name="shortcut_batch_normalization_%d" % stage
        )(residual)
        out = downsample(out, pool_type=pool_type, sort=sort, stage=stage)
        out = tf.keras.layers.Add()([out, residual])
        out = tf.keras.activations.relu(out)
    else:
        out = tf.keras.activations.relu(out)
        out = downsample(out, pool_type=pool_type, sort=sort, stage=stage)
    if pool_type is not None:
        out = tf.keras.layers.Conv1D(
            filters=2 * filters,
            kernel_size=1,
            strides=1,
            padding="same",
            name="1_1_conv_%d" % stage,
        )(out)
        out = tf.keras.layers.BatchNormalization(
            name="1_1_batch_normalization_%d" % stage
        )(out)
    return out


def downsample(inputs, pool_type="max", sort=True, stage=1):
    if pool_type == "max":
        out = tf.keras.layers.MaxPooling1D(
            pool_size=3, strides=2, padding="same", name="pool_%d" % stage
        )(inputs)
    elif pool_type == "k_max":
        k = int(inputs._keras_shape[1] / 2)
        out = KMaxPooling(k=k, sort=sort, name="pool_%d" % stage)(inputs)
    elif pool_type == "conv":
        out = tf.keras.layers.Conv1D(
            filters=inputs._keras_shape[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            name="pool_%d" % stage,
        )(inputs)
        out = tf.keras.layers.BatchNormalization()(out)
    elif pool_type is None:
        out = inputs
    else:
        raise ValueError("unsupported pooling type!")
    return out


def VDCNN(
    num_classes,
    depth=9,
    sequence_length=1024,
    embedding_dim=16,
    shortcut=False,
    pool_type="max",
    sort=True,
    use_bias=False,
    input_tensor=None,
):
    if depth == 9:
        num_conv_blocks = (1, 1, 1, 1)
    elif depth == 17:
        num_conv_blocks = (2, 2, 2, 2)
    elif depth == 29:
        num_conv_blocks = (5, 5, 2, 2)
    elif depth == 49:
        num_conv_blocks = (8, 8, 5, 3)
    else:
        raise ValueError("unsupported depth for VDCNN.")

    inputs = tf.keras.Input(shape=(sequence_length,), name="inputs")
    embedded_chars = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=embedding_dim
    )(inputs)
    out = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=1, padding="same", name="temp_conv"
    )(embedded_chars)

    # Convolutional Block 64
    for _ in range(num_conv_blocks[0] - 1):
        out = identity_block(
            out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    out = conv_block(
        out,
        filters=64,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=shortcut,
        pool_type=pool_type,
        sort=sort,
        stage=1,
    )

    # Convolutional Block 128
    for _ in range(num_conv_blocks[1] - 1):
        out = identity_block(
            out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    out = conv_block(
        out,
        filters=128,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=shortcut,
        pool_type=pool_type,
        sort=sort,
        stage=2,
    )

    # Convolutional Block 256
    for _ in range(num_conv_blocks[2] - 1):
        out = identity_block(
            out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    out = conv_block(
        out,
        filters=256,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=shortcut,
        pool_type=pool_type,
        sort=sort,
        stage=3,
    )

    # Convolutional Block 512
    for _ in range(num_conv_blocks[3] - 1):
        out = identity_block(
            out, filters=512, kernel_size=3, use_bias=use_bias, shortcut=shortcut
        )
    out = conv_block(
        out,
        filters=512,
        kernel_size=3,
        use_bias=use_bias,
        shortcut=False,
        pool_type=None,
        stage=4,
    )

    # k-max pooling with k = 8
    out = KMaxPooling(k=8, sort=True)(out)
    out = tf.keras.layers.Flatten()(out)

    # Dense Layers
    out = tf.keras.layers.Dense(2048, activation="relu")(out)
    out = tf.keras.layers.Dense(2048, activation="relu")(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    if input_tensor is not None:
        inputs = tf.keras.get_source_inputs(input_tensor)
    else:
        inputs = inputs

    # Create model.
    model = tf.keras.Model(inputs=inputs, outputs=out, name="VDCNN")
    return model


if __name__ == "__main__":
    model = VDCNN(10, depth=9, shortcut=False, pool_type="max")
    model.summary()
