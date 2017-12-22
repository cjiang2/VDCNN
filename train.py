import os
import numpy as np
import datetime
import tensorflow as tf
from data_helper import *

# State which model to use here
from vdcnn import VDCNN

# Parameters settings
# Data loading params
tf.flags.DEFINE_string("database_path", "ag_news_csv/", "Path for the dataset to be used.")

# Model Hyperparameters
tf.flags.DEFINE_float("weight_decay", 1e-4, "Weight decay ratio (default: 1e-4)")
tf.flags.DEFINE_integer("sequence_max_length", 1024, "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("downsampling_type", "k-maxpool", "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')")
tf.flags.DEFINE_string("num_layers", "2,2,2,2", "Comma-separated No. of blocks, use either four of '2,2,2,2', '4,4,4,4', '10,10,4,4' or '16,16,10,6'")
tf.flags.DEFINE_boolean("use_he_uniform", True, "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("use_bias", False, "Use bias (default: False)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_boolean("enable_tensorboard", True, "Enable Tensorboard (default: True)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr, value))
print("")

# Data Preparation
# Load data
print("Loading data...")
data_helper = data_helper(FLAGS.sequence_max_length)
train_data, train_label, test_data, test_label = data_helper.load_dataset(FLAGS.database_path)
num_batches_per_epoch = int((len(train_data)-1)/FLAGS.batch_size) + 1
print("Loading data succees...")

# ConvNet
acc_list = [0]
sess = tf.Session()
cnn = VDCNN(num_classes=train_label.shape[1], 
	sequence_max_length=FLAGS.sequence_max_length, 
	downsampling_type=FLAGS.downsampling_type,
	weight_decay=FLAGS.weight_decay,
	use_he_uniform=FLAGS.use_he_uniform,
	use_bias=FLAGS.use_bias,
	num_layers=list(map(int, FLAGS.num_layers.split(","))))

# Optimizer and LR Decay
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
	gradients, variables = zip(*optimizer.compute_gradients(cnn.loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
	train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# Initialize Graph
sess.run(tf.global_variables_initializer())

# Train Step and Test Step
def train_step(x_batch, y_batch):
	"""
	A single training step
	"""
	feed_dict = {cnn.input_x: x_batch, 
				 cnn.input_y: y_batch, 
				 cnn.is_training: True}
	_, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
	time_str = datetime.datetime.now().isoformat()
	print("{}: Step {}, Epoch {}, Loss {:g}, Acc {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, accuracy))
	#if step%FLAGS.evaluate_every == 0 and FLAGS.enable_tensorboard:
	#	summaries = sess.run(train_summary_op, feed_dict)
	#	train_summary_writer.add_summary(summaries, global_step=step)

def test_step(x_batch, y_batch):
	"""
	Evaluates model on a dev set
	"""
	feed_dict = {cnn.input_x: x_batch, 
				 cnn.input_y: y_batch, 
				 cnn.is_training: False}
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
		test_batches = data_helper.batch_iter(list(zip(test_data, test_label)), FLAGS.batch_size, 1, shuffle=False)
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
		print("{}: Current Max Acc {:g} in Iteration {}".format(time_str, max(acc_list), int(acc_list.index(max(acc_list))*FLAGS.evaluate_every)))