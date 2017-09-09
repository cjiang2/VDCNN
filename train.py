import os
import numpy as np
import datetime
import tensorflow as tf
import data_helper

# State which model to use here
from vdcnn_9 import VDCNN

# Parameters settings
# Data loading params
tf.flags.DEFINE_string("database_path", "ag_news_csv/", "Path for the dataset to be used.")

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("sequence_max_length", 1014, "Sequence Max Length (default: 1014)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value))
print("")

# Data Preparation
# Load data
print("Loading data...")
train_data, train_label, test_data, test_label = data_helper.load_dataset(FLAGS.database_path)
print("Loading data succees...")

# ConvNet
acc_list = [0]
sess = tf.Session()
cnn = VDCNN(num_classes=len(train_label[0]), 
	l2_reg_lambda=FLAGS.l2_reg_lambda, 
	sequence_max_length=FLAGS.sequence_max_length, 
	num_quantized_chars=69, 
	embedding_size=16)

# Optimizer and LR Decay
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
	lr_decay_fn = lambda lr, global_step : tf.train.exponential_decay(lr, global_step, 100, 0.95, staircase=True) 
	train_op = tf.contrib.layers.optimize_loss(loss=cnn.loss, global_step=global_step, clip_gradients=4.0,
		learning_rate=FLAGS.learning_rate, optimizer=lambda lr: optimizer, update_ops=update_ops, learning_rate_decay_fn=lr_decay_fn)

# Initialize Graph
sess.run(tf.global_variables_initializer())

# Train Step and Test Step
def train_step(x_batch, y_batch):
	"""
	A single training step
	"""
	feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob, cnn.is_training: True}
	_, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
	time_str = datetime.datetime.now().isoformat()
	print("{}: Step {}, Loss {:g}, Acc {:g}".format(time_str, step, loss, accuracy))

def test_step(x_batch, y_batch):
	"""
	Evaluates model on a dev set
	"""
	feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0, cnn.is_training: False}
	loss, preds = sess.run([cnn.loss, cnn.predictions], feed_dict)
	time_str = datetime.datetime.now().isoformat()
	return preds, loss

# Generate batches
train_batches = data_helper.batch_iter(list(zip(train_data, train_label)), FLAGS.batch_size, FLAGS.num_epochs)

# Training loop. For each batch...
for train_batch in train_batches:
	x_batch, y_batch = zip(*train_batch)
	train_step(x_batch, y_batch)
	current_step = tf.train.global_step(sess, global_step)
	# Testing loop
	if current_step % FLAGS.evaluate_every == 0:
		print("\nEvaluation:")
		i = 0
		index = 0
		sum_loss = 0
		test_batches = data_helper.batch_iter(list(zip(test_data, test_label)), FLAGS.batch_size, 1)
		y_preds = np.ones(shape=len(test_label), dtype=np.int)
		for test_batch in test_batches:
			x_test_batch, y_test_batch = zip(*test_batch)
			preds, test_loss = test_step(x_test_batch, y_test_batch)
			sum_loss += test_loss
			res = np.absolute(preds - np.argmax(y_test_batch, axis=1))
			y_preds[index:index+len(res)] = res
			i += 1
			index += len(res)

		time_str = datetime.datetime.now().isoformat()
		acc = np.count_nonzero(y_preds==0)/len(y_preds)
		acc_list.append(acc)
		print("{}: Evaluation Summary, Loss {:g}, Acc {:g}".format(time_str, sum_loss/i, acc))
		print("{}: Current Max Acc {:g} with in Iteration {}".format(time_str, max(acc_list), int(acc_list.index(max(acc_list))*FLAGS.evaluate_every)))