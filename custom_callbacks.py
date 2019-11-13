"""
Some custom callback function to strengthen up training code and tensorboard
"""
import datetime

import tensorflow as tf


class LossHistory(tf.keras.callbacks.Callback):
    """
    Record loss history by step in Tensorboard
    """

    def __init__(self, model, tensorboard, names=None):
        self.model = model
        self.tensorboard = tensorboard
        if names is None:
            self.names = ["acc", "loss"]
        else:
            self.names = names
        self.step = 0

    def on_train_begin(self, logs={}):
        self.step = 0

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for name in self.names:
            writer = tf.summary.create_file_writer("/tmp/mylogs")
            with writer.as_default():
                tag = name + "_step"
                tf.summary.scalar(tag, logs[name], step=self.step)
                writer.flush()


class EvaluateStep(tf.keras.callbacks.Callback):
    """
    Custom callback function to enable evaluation per step
    """

    def __init__(
        self, model, checkpointer, tensorboard, evaluate_every, batch_size, x_dev, y_dev
    ):
        self.model = model
        self.evaluate_every = evaluate_every
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.batch_size = batch_size
        self.checkpointer = checkpointer
        self.tensorboard = tensorboard
        self.max_step = 0
        self.step = 0
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.step = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        if self.step % self.evaluate_every == 0:
            logs = self.model.evaluate(
                x=self.x_dev, y=self.y_dev, batch_size=self.batch_size, verbose=0
            )
            if self.checkpointer.monitor_op(logs[1], self.checkpointer.best):
                self.checkpointer.best = logs[1]
                self.max_step = self.step
                path = "checkpoints/vdcnn_weights_val_acc_%0.4f.h5" % (
                    self.checkpointer.best
                )
                if self.checkpointer.save_weights_only:
                    self.model.save_weights(path, overwrite=True)
                else:
                    self.model.save(path, overwrite=True)
                time_str = datetime.datetime.now().isoformat()
                print()
                print(
                    "{}: Saving model with val_acc {:g}, at step {}, epoch {}.".format(
                        time_str, self.checkpointer.best, self.max_step, self.epoch + 1
                    )
                )
                print()
            if self.tensorboard is not None:
                names = ["val_loss_step", "val_acc_step"]
                for idx, val in enumerate(names):
                    writer = tf.summary.create_file_writer("/tmp/mylogs")
                    with writer.as_default():
                        tf.summary.scalar(val, logs[idx], step=self.step)
                        writer.flush()
