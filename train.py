import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from vdcnn import *
from utils import *

# --------------------
# Hyperparameters
# --------------------
MAXLEN = 1014
DEPTH = 9
EMBED_DIM = 16
SHORTCUT = True
POOL_TYPE = 'k_max'
PROJ_TYPE = 'identity'
USE_BIAS = True

BATCH_SIZE = 128
SHUFFLE_BUFFER = 1024
LR = 1e-2
EPOCHS = 20
CLIP_NORM = 7.0

DATASET_NAME = 'ag_news'

CHECKPOINT_PATH = "./checkpoints"
DISPLAY_EVERY = 20

# --------------------
# Helper Functions
# --------------------
def prepare_data(dataset_name='ag_news', 
                 split='train'):
    shuffle_files = True if split == 'train' else False

    if dataset_name == 'ag_news':
        ds = tfds.load('ag_news_subset', split=split, shuffle_files=shuffle_files)
        num_classes = 4

    return ds, num_classes

@tf.function
def train_step(inputs, labels):
    # Forward pass
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = loss_object(labels, logits)

    # Backward
    gradients = tape.gradient(loss, model.trainable_variables)
    if CLIP_NORM is not None:
        # Gradient clipping to stabilize training
        gradients = [tf.clip_by_norm(grad, CLIP_NORM) for grad in gradients]

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Metrics
    preds = tf.nn.softmax(logits)
    train_loss(loss)
    train_accuracy(labels, preds) # Train accuracy

@tf.function
def test_step(inputs, labels):
    logits = model(inputs, training=False)
    t_loss = loss_object(labels, logits)

    preds = tf.nn.softmax(logits)
    test_loss(t_loss)
    test_accuracy(labels, preds)

# --------------------
# Training
# --------------------
# Dataset for training
ds_train, num_classes = prepare_data(DATASET_NAME, 'train')
ds_train = ds_train.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

ds_test, _ = prepare_data(DATASET_NAME, 'test')
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Tokenizer
tokenizer = Tokenizer()

# Model
model = VDCNN(num_classes=num_classes, 
              depth=DEPTH,
              vocab_size=69,
              seqlen=MAXLEN,
              embed_dim=EMBED_DIM,
              shortcut=SHORTCUT,
              pool_type=POOL_TYPE,
              proj_type=PROJ_TYPE,
              use_bias=USE_BIAS)

# Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=LR, 
                                    momentum=0.0)
    
# Loss and Metrices
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

# Checkpoint
ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=None)

# Loop
step = 0
for epoch in range(EPOCHS):
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    # Train Loop
    for batch in ds_train:
        texts = batch['description'].numpy()
        labels = tf.keras.utils.to_categorical(batch['label'], num_classes=num_classes)

        # Convert to sequence HERE
        # Shady bypass of tfds in favor of custom data_op
        inputs = np.array([tokenizer.text_to_sequence(text.decode('ascii')) for text in texts])
        inputs = tf.convert_to_tensor(inputs)

        # One train step
        train_step(inputs, labels)

        if step % DISPLAY_EVERY == 0:
            print(f'Epoch {epoch + 1}, ' 
                  f'Step {step}, '
                  f'Loss: {train_loss.result()}, '
                  f'Accuracy: {train_accuracy.result() * 100}')

        step += 1
    
    # Test Loop
    for batch_test in ds_test:
        texts = batch_test['description'].numpy()
        labels = tf.keras.utils.to_categorical(batch_test['label'], num_classes=num_classes)

        # Convert to sequence HERE
        # Shady bypass of tfds in favor of custom data_op
        inputs = np.array([tokenizer.text_to_sequence(text.decode('ascii')) for text in texts])
        inputs = tf.convert_to_tensor(inputs)

        # One train step
        test_step(inputs, labels)

    print(f'Epoch {epoch + 1}, ' 
          f'Test Loss: {test_loss.result()}, '
          f'Test Accuracy: {test_accuracy.result() * 100}')

    # Save model every epoch
    ckpt_manager.save()
