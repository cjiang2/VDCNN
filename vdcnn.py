import tensorflow as tf
import numpy as np

# weights initializers
he_normal = tf.keras.initializers.he_normal()

def Convolutional_Block(inputs, num_layers, num_filters, name, is_training, weight_decay, use_bias=False):
    # Convolutional Block which contains 2 Conv layers
    with tf.variable_scope("conv_block_%s" % name):
        filter_shape = [3, inputs.get_shape()[2], num_filters]
        w = tf.get_variable(name='W_1', shape=filter_shape, 
            initializer=he_normal,
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        conv = tf.nn.conv1d(inputs, w, stride=1, padding="SAME")
        if use_bias:
            b = tf.get_variable(name='b_1', shape=[num_filters], 
                initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
        bn = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, 
                                           center=True, scale=True, training=is_training)
        out = tf.nn.relu(bn)

        for i in range(2, num_layers+1):
            filter_shape = [3, out.get_shape()[2], num_filters]
            w = tf.get_variable(name='W_'+str(i), shape=filter_shape, 
                initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv1d(out, w, stride=1, padding="SAME")
            if use_bias:
                b = tf.get_variable(name='b_'+str(i), shape=[num_filters], 
                    initializer=tf.constant_initializer(0.0))
                conv = tf.nn.bias_add(conv, b)
            bn = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, 
                                               center=True, scale=True, training=is_training)
            out = tf.nn.relu(bn)
    return out

class VDCNN():
    def __init__(self, num_classes, weight_decay=1e-4, sequence_max_length=1024, num_quantized_chars=69, embedding_size=16, num_layers=[2,2,2,2],
                 use_k_max_pooling=False, use_bias=False):
        # input tensors
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training =  tf.placeholder(tf.bool)

        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #self.embedding_W = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0),name="embedding_W")
            self.embedding_W = tf.get_variable(name='lookup_W', shape=[num_quantized_chars, embedding_size], initializer=tf.keras.initializers.he_uniform())
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            print(self.embedded_characters.get_shape())

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv") as scope: 
            filter_shape = [3, embedding_size, 64]
            w = tf.get_variable(name='W_1', shape=filter_shape, 
                initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            out = tf.nn.conv1d(self.embedded_characters, w, stride=1, padding="SAME")
            if use_bias:
                b = tf.get_variable(name='b_1', shape=[64], 
                        initializer=tf.constant_initializer(0.0))
                out = tf.nn.bias_add(out, b)
            self.temp_conv = tf.nn.relu(out)
            print(self.temp_conv.get_shape())

        # all convolutional blocks
        self.conv_block_1 = Convolutional_Block(self.temp_conv, num_layers=num_layers[0], num_filters=64, name='1', weight_decay=weight_decay, is_training=self.is_training)
        self.pool1 = tf.layers.max_pooling1d(inputs=self.conv_block_1, pool_size=3, strides=2, padding='same', name='pool1')
        print(self.pool1.get_shape())

        self.conv_block_2 = Convolutional_Block(self.pool1, num_layers=num_layers[1], num_filters=128, name='2', weight_decay=weight_decay, is_training=self.is_training)
        self.pool2 = tf.layers.max_pooling1d(inputs=self.conv_block_2, pool_size=3, strides=2, padding='same', name='pool2')
        print(self.pool2.get_shape())

        self.conv_block_3 = Convolutional_Block(self.pool2, num_layers=num_layers[2], num_filters=256, name='3', weight_decay=weight_decay, is_training=self.is_training)
        self.pool3 = tf.layers.max_pooling1d(inputs=self.conv_block_3, pool_size=3, strides=2, padding='same', name='pool3')
        print(self.pool3.get_shape())

        self.conv_block_4 = Convolutional_Block(self.pool3, num_layers=num_layers[3], num_filters=512, name='4', weight_decay=weight_decay, is_training=self.is_training)
        print(self.conv_block_4.get_shape())

        # Transpose since top_k works on the last dimension
        transposed = tf.transpose(self.conv_block_4, [0,2,1])
        # Extract 8 most features as mentioned in paper
        self.k_pooled = tf.nn.top_k(transposed, k=8, name='k_pool', sorted=False)
        print(transposed.get_shape(), self.k_pooled[0].get_shape())
        self.flatten = tf.reshape(self.k_pooled[0], (-1, 512*8))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(0.0))
            out = tf.matmul(self.flatten, w) + b
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(0.0))
            out = tf.matmul(self.fc1, w) + b
            self.fc2 = tf.nn.relu(out)

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [self.fc2.get_shape()[1], num_classes], initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            self.fc3 = tf.matmul(self.fc2, w) + b

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc3, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.input_y)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(losses) + sum(regularization_losses)
            # + weight_decay * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")