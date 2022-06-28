import tensorflow as tf
from tensorflow.keras import Model, layers

N_BLOCKS = {9: (1, 1, 1, 1),
            17: (2, 2, 2, 2),
            29: (5, 5, 2, 2),
            49:(8, 8, 5, 3)}

class KMaxPooling(layers.Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, 
                 k=None, 
                 sorted=False):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.sorted = sorted

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, input_shape[2])

    def call(self, 
             inputs):
        if self.k is None:
            k = int(tf.round(inputs.shape[1] / 2))
        else:
            k = self.k

        # Swap last two dimensions since top_k will be applied along the last dimension
        shifted_inputs = tf.transpose(inputs, [0, 2, 1])
        
        # Extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_inputs, k=k, sorted=self.sorted)[0]
        
        # return flattened output
        return tf.transpose(top_k, [0, 2, 1])

class Pooling(layers.Layer):
    """Wrapper for different pooling operations.
    Including maxpooling and k-maxpooling.
    """
    def __init__(self, 
                 pool_type='max',
                 name=None):
        super(Pooling, self).__init__(name=name)
        assert pool_type in ['max', 'k_max']
        self.pool_type = pool_type

        if pool_type == 'max':
            self.pool = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')
        elif pool_type == 'k_max':
            self.pool = KMaxPooling()
        
    def call(self, 
             x):
        return self.pool(x)

class ZeroPadding(layers.Layer):
    def __init__(self, 
                 values,
                 name=None):
        super(ZeroPadding, self).__init__(name=name)
        self.values = values

    def call(self, 
             x):
        x = tf.pad(x, [[0, 0], [0, 0], [self.values[0], self.values[1]]], 
                   mode='CONSTANT', constant_values=0)
        return x

class Conv1D_BN(layers.Layer):
    """A stack of conv 1x1 and BatchNorm.
    """
    def __init__(self, 
                 filters,
                 kernel_size=3,
                 strides=2,
                 padding='same',
                 use_bias=True,
                 name=None):
        super(Conv1D_BN, self).__init__(name=name)
        self.filters = filters
        self.use_bias = use_bias
        self.conv = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                  kernel_initializer='he_normal')
        self.bn = layers.BatchNormalization()

    def call(self, 
             x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class ConvBlock(layers.Layer):
    """Conv block with downsampling.
    1x1 conv to increase dimensions.
    """
    def __init__(self, 
                 filters, 
                 kernel_size=3,
                 use_bias=True,
                 shortcut=True,
                 pool_type=None,
                 proj_type=None,
                 name=None,
                 ):
        super(ConvBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.proj_type = proj_type

        # Deal with downsample and pooling
        assert pool_type in ['max', 'k_max', 'conv', None]
        if pool_type is None:
            strides = 1
            self.pool = None
            self.downsample = None

        elif pool_type == 'conv':
            strides = 2     # Convolutional pooling with stride 2
            self.pool = None
            if shortcut:
                self.downsample = Conv1D_BN(filters, 3, strides=2, padding='same', use_bias=use_bias)
        
        else:
            strides = 1
            self.pool = Pooling(pool_type)
            if shortcut:
                self.downsample = Conv1D_BN(filters, 3, strides=2, padding='same', use_bias=use_bias)

        self.conv1 = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, 
                                   kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias,
                                   kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()

        assert proj_type in ['identity', 'conv', None]
        if shortcut:
            if proj_type == 'conv':
                # 1x1 conv for projection
                self.proj = Conv1D_BN(filters*2, 1, strides=1, padding='same', use_bias=use_bias)

            elif proj_type == 'identity':
                # Identity using zero padding
                self.proj = ZeroPadding([int(filters // 2), filters - int(filters // 2)])

    def call(self, 
             x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.pool is not None:
            out = self.pool(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        out = tf.nn.relu(out)

        if self.proj_type is not None and self.shortcut:
            out = self.proj(out)

        return out

class VDCNN(Model):
    """Model codebase for VDCNN.
    Args:
        num_classes: No. classes for classification task.
        depth: depth of VDCNN, one of [9, 17, 29, 49].
        seqlen: Sequence length.
        embed_dim: dim for character embeddings.
        shortcut: Use skip connections.
        pool_type: Pooling operations to be used, one of ['max', 'k_max', 'conv'].
        proj_type: Operation to increase dim for dotted skip connection, one of ['identity', 'conv'].
        use_bias: Use bias for all layers or not.
        logits: If False, return softmax probs.
    """
    def __init__(self, 
                 num_classes,
                 depth=9, 
                 vocab_size=69,
                 seqlen=None,
                 embed_dim=16,
                 shortcut=True, 
                 pool_type='max',
                 proj_type='conv',
                 use_bias=True,
                 logits=True):
        super(VDCNN, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.vocab_size = vocab_size
        self.seqlen = seqlen
        self.embed_dim = embed_dim
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.proj_type = proj_type
        self.use_bias = use_bias
        self.logits = logits

        assert pool_type in ['max', 'k_max', 'conv']
        assert proj_type in ['conv', 'identity']
        self.n_blocks = N_BLOCKS[depth]

        self.embed_char = layers.Embedding(vocab_size, embed_dim, input_length=seqlen)
        self.conv = layers.Conv1D(64, 3, strides=1, padding='same', use_bias=use_bias, 
                                  kernel_initializer='he_normal')

        # Convolutional Block 64
        self.conv_block_64 = []
        for _ in range(self.n_blocks[0] - 1):
            self.conv_block_64.append(ConvBlock(64, 3, use_bias, shortcut))
        self.conv_block_64.append(ConvBlock(64, 3, use_bias, shortcut, pool_type=pool_type, proj_type=proj_type))

        # Convolutional Block 128
        self.conv_block_128 = []
        for _ in range(self.n_blocks[1] - 1):
            self.conv_block_128.append(ConvBlock(128, 3, use_bias, shortcut))
        self.conv_block_128.append(ConvBlock(128, 3, use_bias, shortcut, pool_type=pool_type, proj_type=proj_type))

        # Convolutional Block 256
        self.conv_block_256 = []
        for _ in range(self.n_blocks[2] - 1):
            self.conv_block_256.append(ConvBlock(256, 3, use_bias, shortcut))
        self.conv_block_256.append(ConvBlock(256, 3, use_bias, shortcut, pool_type=pool_type, proj_type=proj_type))

        # Convolutional Block 512
        self.conv_block_512 = []
        for _ in range(self.n_blocks[3] - 1):
            self.conv_block_512.append(ConvBlock(512, 3, use_bias, shortcut))
        self.conv_block_512.append(ConvBlock(512, 3, use_bias, shortcut, pool_type=None, proj_type=None))

        self.k_maxpool = KMaxPooling(k=8)
        self.flatten = layers.Flatten()

        # Dense layers
        self.fc1 = layers.Dense(2048, activation='relu')
        self.fc2 = layers.Dense(2048, activation='relu')
        self.out = layers.Dense(num_classes)

    def call(self,
             x):
        x = self.embed_char(x)
        #print('embed:', x.shape)
        x = self.conv(x)
        #print('conv:', x.shape)

        for l in self.conv_block_64:
            x = l(x)
        #print('conv_block_64:', x.shape)

        for l in self.conv_block_128:
            x = l(x)
        #print('conv_block_128:', x.shape)

        for l in self.conv_block_256:
            x = l(x)
        #print('conv_block_256:', x.shape)

        for l in self.conv_block_512:
            x = l(x)
        #print('conv_block_512:', x.shape)

        x = self.k_maxpool(x)
        #print('k_maxpool_8:', x.shape)
        x = self.flatten(x)
        #print('flatten:', x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        out = self.out(x)
        #print('out:', out.shape)

        if self.logits:
            return out
        
        return tf.nn.softmax(out)

if __name__ == "__main__":
    x = tf.zeros([4, 1014])
    model = VDCNN(10, depth=9, shortcut=True, pool_type='max', proj_type='identity')
    out = model(x)
    model.summary()

    print()
    model = VDCNN(10, depth=17, shortcut=True, pool_type='k_max', proj_type='identity')
    out = model(x)
    model.summary()

    print()
    model = VDCNN(10, depth=29, shortcut=False, pool_type='max', proj_type='conv')
    out = model(x)
    model.summary()

    print()
    model = VDCNN(10, depth=49, shortcut=True, pool_type='conv', proj_type='conv')
    out = model(x)
    model.summary()
