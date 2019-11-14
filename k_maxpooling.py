import tensorflow as tf


class KMaxPooling(tf.keras.layers.Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, sort=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.k = k
        self.sort = sort

    def get_config(self):
        super().get_config()

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k, input_shape[2]

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_inputs = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.math.top_k(shifted_inputs, k=self.k, sorted=self.sort)[0]

        # return flattened output
        return tf.transpose(top_k, [0, 2, 1])
