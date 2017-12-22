import tensorflow as tf
import numpy as np
import math

# weights initializers
he_normal = tf.keras.initializers.he_normal()

def Convolutional_Block(inputs, num_filters, name, is_training, weight_decay, use_bias=False, optional_shortcut=False):
    print("-"*20)
    print("Convolutional Block", str(num_filters), name)
    print("-"*20)
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape, 
                    initializer=he_normal,
                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                if use_bias:
                    b = tf.get_variable(name='b', shape=[num_filters], 
                        initializer=tf.constant_initializer(0.0))
                    inputs = tf.nn.bias_add(inputs, b)
                inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5, 
                                                center=True, scale=True, training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Conv1D:", inputs.get_shape())
    print("-"*20)
    return inputs

# Three types of downsampling methods described by paper
def downsampling(inputs, downsampling_type, name):
    # k-maxpooling
    if downsampling_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        k_pooled = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        return tf.transpose(k_pooled, [0,2,1])
    # Linear
    elif downsampling_type=='linear':
        with tf.variable_scope(name):
            filter_shape = [3, inputs.get_shape()[2], inputs.get_shape()[2]]
            w = tf.get_variable(name='W', shape=filter_shape, 
                                initializer=he_normal,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            return tf.nn.conv1d(inputs, w, stride=2, padding="SAME")
    # Maxpooling
    else:
        return tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)

class VDCNN():
    def __init__(self, num_classes, sequence_max_length=1024, num_quantized_chars=69, embedding_size=16, 
                 num_layers=[2,2,2,2], num_filters=[64,128,256,512], downsampling_type='maxpool', weight_decay=1e-4,
                 use_bias=False, use_he_uniform=False, optional_shortcut=True):

        # Make sure legit inputs for num_layers and num_filters
        assert(len(num_layers)==len(num_filters))

        # input tensors
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training =  tf.placeholder(tf.bool)

        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if use_he_uniform:
                self.embedding_W = tf.get_variable(name='lookup_W', shape=[num_quantized_chars, embedding_size], initializer=tf.keras.initializers.he_uniform())
            else:
                self.embedding_W = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0),name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            print("-"*20)
            print("Embedded Lookup:", self.embedded_characters.get_shape())
            print("-"*20)

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv") as scope: 
            filter_shape = [3, embedding_size, 64]
            w = tf.get_variable(name='W_1', shape=filter_shape, 
                initializer=he_normal,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            inputs = tf.nn.conv1d(self.embedded_characters, w, stride=1, padding="SAME")
            if use_bias:
                b = tf.get_variable(name='b_1', shape=[64], 
                        initializer=tf.constant_initializer(0.0))
                inputs = tf.nn.bias_add(inputs, b)
            #inputs = tf.nn.relu(inputs)
        print("Temp Conv", inputs.get_shape())

        # all convolutional blocks
        for i in range(len(num_layers)):
            for j in range(num_layers[i]):
                inputs = Convolutional_Block(inputs=inputs, num_filters=num_filters[i], is_training=self.is_training, 
                                             weight_decay=weight_decay, use_bias=False, name=str(j+1))
            if i < len(num_layers) - 1:
                inputs = downsampling(inputs, downsampling_type=downsampling_type, name='pool'+str(j+1))
                print("Pooling:", inputs.get_shape())
                print("-"*20)

        # Extract 8 most features as mentioned in paper
        k_pooled = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=8, name='k_pool', sorted=False)[0]
        print("8-maxpooling:", k_pooled.get_shape())
        self.flatten = tf.reshape(k_pooled, (-1, 512*8))

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

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")