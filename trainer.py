""" Trains a TensorFlow model
Example:
$ python trainer.py --checkpoint_path ./checkpoints --data_path ./tfrecords
"""

import tensorflow as tf
import numpy as np
import os, glob
import argparse
from network import model
#from nets import mobilenet_v1 #####################################################

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class TFModelTrainer:

    def __init__(self, checkpoint_path, data_path):
        self.checkpoint_path = checkpoint_path

        # set training parameters #################################################
        self.learning_rate = 0.002
        self.training_epochs = 8000
        self.save_iter = 100
        self.val_iter = 800
        self.log_iter = 100
        self.batch_size = 32

        # set up data layer
        self.training_filenames = glob.glob(os.path.join(data_path, 'train_*.tfrecord'))
        self.validation_filenames = glob.glob(os.path.join(data_path, 'test_*.tfrecord'))
        self.iterator, self.filenames = self.data_layer()
        self.num_val_samples = 1000
        self.num_classes = 3
        self.image_height = 352
        self.image_width = 288
        self.image_size = 224

    def preprocess_image(self, image_string):
        image = tf.image.decode_png(image_string, channels=3)

        # flip for data augmentation
        image = tf.image.random_flip_left_right(image) ############################

        # normalize image to [0 : +1]
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image

    def parse_tfrecord(self, example_proto): #####################################
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            'label': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        image = parsed_features['image']
        label = parsed_features['label']
        image = self.preprocess_image(image)
        return image, label

    def data_layer(self, num_threads=8, prefetch_buffer=100):
        with tf.variable_scope('data'):
            filenames = tf.placeholder(tf.string, shape=[None])
            dataset = tf.data.TFRecordDataset(filenames) ##########################
            dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=num_threads)
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(prefetch_buffer)
            iterator = dataset.make_initializable_iterator()
        return iterator, filenames

    def loss_functions(self, logits, labels):
        with tf.variable_scope('loss'):
            target_prob = tf.one_hot(labels, self.num_classes)
            tf.losses.softmax_cross_entropy(target_prob, logits)
            total_loss = tf.losses.get_total_loss() #include regularization loss
        return total_loss

    def optimizer(self, total_loss, global_step):
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
        return optimizer

    def performance_metric(self, logits, labels):
        with tf.variable_scope("performance_metric"):
            predicted = tf.argmax(logits, axis=1)
            labels = tf.cast(labels, tf.int64)
            corrects = tf.equal(predicted, labels)
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
        return accuracy

    def train(self):
        # iteration number
        global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='iter_number')

        # training graph
        images, labels = self.iterator.get_next()
        images = tf.image.resize_bilinear(images, (self.image_height, self.image_width))
        logits = model(images, num_classes=self.num_classes)
        loss = self.loss_functions(logits, labels)
        optimizer = self.optimizer(loss, global_step)
        accuracy = self.performance_metric(logits, labels)

        # Create a summary operation to log the progress of the network
        with tf.variable_scope('logging'):
            # summary placeholders
            streaming_loss_p = tf.placeholder(tf.float32)
            accuracy_p = tf.placeholder(tf.float32)

            summary_op_train = tf.summary.scalar('streaming_loss', streaming_loss_p)
            summary_op_test = tf.summary.scalar('accuracy', accuracy_p)

        # Create ops to save and restore all the variables.
        with tf.variable_scope('saving'):
            saver = tf.train.Saver(max_to_keep=None)  # keep all checkpoints

        # Initialize a sessionion so that we can run TensorFlow operations
        with tf.Session() as session:

            # Run the global variable initializer to initialize all variables and layers of the neural network
            session.run(tf.global_variables_initializer())
            session.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})

            # Create log file writers to record training progress.
            # We'll store training and testing log data separately.
            training_writer = tf.summary.FileWriter("./logs/training", session.graph)
            testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)
            # writer = tf.summary.FileWriter(self.checkpoint_path, session.graph)

            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)

            # resume training if a checkpoint exists
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                print('Loaded parameters from {}'.format(ckpt.model_checkpoint_path))

            initial_step = global_step.eval()

            # One epoch is one full run through the training data set.
            streaming_loss = 0
            for epoch in range(initial_step, self.training_epochs + 1):
                # Run the optimizer over and over to train the network.
                session.run(optimizer)
                # Get the current accuracy scores by running the "cost" operation on the training and test data sets
                loss_batch = session.run(loss)

                # Log summary
                streaming_loss += loss_batch
                if epoch % self.log_iter == self.log_iter - 1:
                    streaming_loss /= self.log_iter
                    print("Epoch: {} - Training Cost: {}".format(epoch + 1, streaming_loss))
                    training_summary = session.run(summary_op_train, feed_dict={streaming_loss_p: streaming_loss})

                    # Write the current training status to the log files (Which we can view with TensorBoard)
                    training_writer.add_summary(training_summary, global_step=epoch)

                    streaming_loss = 0

                # save model
                if epoch % self.save_iter == self.save_iter - 1:
                    save_path = saver.save(session, os.path.join(self.checkpoint_path, 'checkpoint'),
                                           global_step=global_step)
                    print("Model saved!")

                # Run validation
                if epoch % self.val_iter == self.val_iter - 1:
                    print("Running validation.")
                    session.run(self.iterator.initializer, feed_dict={self.filenames: self.validation_filenames})

                    validation_accuracy = 0
                    for j in range(self.num_val_samples // self.batch_size):
                        acc_batch = session.run(accuracy)
                        validation_accuracy += acc_batch
                    validation_accuracy /= j

                    print("Accuracy: {}".format(validation_accuracy))

                    testing_summary = session.run(summary_op_test, feed_dict={accuracy_p: validation_accuracy})
                    testing_writer.add_summary(testing_summary, global_step=epoch)

                    session.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})

            training_writer.close()
            testing_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--data_path', type=str, default='./tfrecords/', help="Path to the TFRecords")
    args = parser.parse_args()
    trainer = TFModelTrainer(args.checkpoint_path, args.data_path)
    trainer.train()

if __name__ == '__main__':
    main()
