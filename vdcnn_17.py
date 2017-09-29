import tensorflow as tf
import numpy as np

# weights initializers
conv_initializer = tf.contrib.keras.initializers.he_normal()
linear_initializer = tf.contrib.keras.initializers.he_normal()

def Convolutional_Block(inputs, num_layers, num_filters, name, is_training):
    # Convolutional Block which contains 2 Conv layers
    with tf.variable_scope("conv_block_%s" % name):
        filter_shape = [3, 1, inputs.get_shape()[3], num_filters]
        w = tf.get_variable(name='W_1', shape=filter_shape, 
            initializer=conv_initializer)
        b = tf.get_variable(name='b_1', shape=[num_filters], 
                initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
        out = tf.nn.relu(batch_norm)

        for i in range(2, num_layers+1):
            filter_shape = [3, 1, out.get_shape()[3], num_filters]
            w = tf.get_variable(name='W_'+str(i), shape=filter_shape, 
                initializer=conv_initializer)
            b = tf.get_variable(name='b_'+str(i), shape=[num_filters], 
                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.nn.bias_add(conv, b)
            batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
            out = tf.nn.relu(batch_norm)
    return out

class VDCNN():
    def __init__(self, num_classes, l2_reg_lambda=0.0005, sequence_max_length=1014, num_quantized_chars=69, embedding_size=16, use_k_max_pooling=False):
        # input tensors
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training =  tf.placeholder(tf.bool)

        # l2 loss
        l2_loss = tf.constant(0.0)

        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_W = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0),name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            self.embedded_characters_expanded = tf.expand_dims(self.embedded_characters, -1, name="embedding_input")

        # First Conv Layer
        with tf.variable_scope("first_conv") as scope: 
            filter_shape = [3, embedding_size, 1, 64]
            w = tf.get_variable(name='W_1', shape=filter_shape, 
                initializer=conv_initializer)
            conv = tf.nn.conv2d(self.embedded_characters_expanded, w, strides=[1, 1, embedding_size, 1], padding="SAME")
            b = tf.get_variable(name='b_1', shape=[64], 
                    initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, b)
            self.first_conv = tf.nn.relu(out)

        # all convolutional blocks
        self.conv_block_1 = Convolutional_Block(self.first_conv, num_layers=4, num_filters=64, name='1', is_training=self.is_training)
        self.pool1 = tf.nn.max_pool(self.conv_block_1, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_1")

        self.conv_block_2 = Convolutional_Block(self.pool1, num_layers=4, num_filters=128, name='2', is_training=self.is_training)
        self.pool2 = tf.nn.max_pool(self.conv_block_2, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_2")

        self.conv_block_3 = Convolutional_Block(self.pool2, num_layers=4, num_filters=256, name='3', is_training=self.is_training)
        self.pool3 = tf.nn.max_pool(self.conv_block_3, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_3")

        self.conv_block_4 = Convolutional_Block(self.pool3, num_layers=4, num_filters=512, name='4', is_training=self.is_training)

        if use_k_max_pooling:
            transposed = tf.transpose(self.conv_block_4, [0,3,2,1])
            self.k_pooled = tf.nn.top_k(transposed, k=8, name='k_pool')
            reshaped = tf.reshape(self.k_pooled[0], (-1, 512*8))
        else:
            self.pool4 = tf.nn.max_pool(self.conv_block_4, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_4")
            shape = int(np.prod(self.pool4.get_shape()[1:]))
            reshaped = tf.reshape(self.pool4, (-1, shape))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [reshaped.get_shape()[1], 2048], initializer= linear_initializer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(0.0))
            l2_loss += tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
            out = tf.matmul(reshaped, w) + b
            self.fc1 = tf.nn.relu(out)
            self.drop1 = tf.nn.dropout(self.fc1, self.dropout_keep_prob, name='drop1') 

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.drop1.get_shape()[1], 2048], initializer= linear_initializer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(0.0))
            l2_loss += tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
            out = tf.matmul(self.drop1, w) + b
            self.fc2 = tf.nn.relu(out)
            self.drop2 = tf.nn.dropout(self.fc2, self.dropout_keep_prob, name='drop2') 

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [self.drop2.get_shape()[1], num_classes], initializer=linear_initializer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            l2_loss += tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
            self.fc3 = tf.matmul(self.drop2, w) + b

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc3, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.fc3, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")