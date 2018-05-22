'''
Some custom callback function to strengthen up training code and tensorboard
'''
import numpy as np
import keras
import tensorflow as tf
import datetime

class loss_history(keras.callbacks.Callback):
    """
    Record loss history by step in Tensorboard
    """
    def __init__(self, model, tensorboard, names=['acc', 'loss']):
        self.model = model
        self.tensorboard = tensorboard
        self.names = names

    def on_train_begin(self, logs={}):
        self.step = 0

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for name in self.names:
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = logs[name]
            summary_value.tag = name+'_step'
            self.tensorboard.writer.add_summary(summary, self.step)
            self.tensorboard.writer.flush()

class evaluate_step(keras.callbacks.Callback):
    """
    Custom callback function to enable evaluation per step
    """
    def __init__(self, model, checkpointer, tensorboard, evaluate_every, batch_size, 
                 x_dev, y_dev):
        self.model = model
        self.evaluate_every = evaluate_every
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.batch_size = batch_size
        self.checkpointer = checkpointer
        self.tensorboard = tensorboard
        self.max_step = 0

    def on_train_begin(self, logs={}):
        self.step = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        if self.step % self.evaluate_every == 0:
            logs = self.model.evaluate(x=self.x_dev, y=self.y_dev, batch_size=self.batch_size, verbose=0)
            if self.checkpointer.monitor_op(logs[1], self.checkpointer.best):
                self.checkpointer.best = logs[1]
                self.max_step = self.step
                path = 'checkpoints/vdcnn_weights_val_acc_%0.4f.h5' % (self.checkpointer.best)
                if self.checkpointer.save_weights_only:
                    self.model.save_weights(path, overwrite=True)
                else:
                    self.model.save(path, overwrite=True)
                time_str = datetime.datetime.now().isoformat()
                print()
                print("{}: Saving model with val_acc {:g}, at step {}, epoch {}.".format(time_str, self.checkpointer.best, self.max_step, self.epoch+1))
                print()
            if self.tensorboard is not None:
                names = ['val_loss_step', 'val_acc_step']
                for i in range(len(names)):
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = logs[i]
                    summary_value.tag = names[i]
                    self.tensorboard.writer.add_summary(summary, self.step)
                    self.tensorboard.writer.flush()