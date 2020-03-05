""" Trains a TensorFlow model
Example:
$ python trainer.py --checkpoint_path ./checkpoints --data_path ./tfrecords
"""

import tensorflow as tf
import os, glob
import argparse
from ImitationArchitecture import model

# Turn off TensorFlow warning messages in program output
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TFModelTrainer:

    def __init__(self, checkpoint_path, data_path):
        self.checkpoint_path = checkpoint_path

        # set training parameters
        self.learning_rate = 0.002
        self.training_epochs = 2500     # 100 epoch
        self.save_iter = 25
        self.val_iter = 100
        self.log_iter = 25
        self.batch_size = 32

        # set up data layer
        self.training_filenames = glob.glob(os.path.join(data_path, 'train_*.tfrecord'))
        self.validation_filenames = glob.glob(os.path.join(data_path, 'test_*.tfrecord'))
        self.iterator, self.filenames = self._data_layer()
        self.num_val_samples = 1000
        self.num_classes = 3
        self.image_height = 352
        self.image_width = 288

    def preprocess_image(self, image_string):
        # Decode the image to get array of its pixels
        image = tf.image.decode_png(image_string, channels=3)
        # Flip for data augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32)
        # Normalize the image to [0 : +1]
        image = image / 255.0
        return image

    def parse_tfrecord(self, example_proto):
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            'label': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        image = parsed_features['image']
        label = parsed_features['label']
        image = self.preprocess_image(image)
        return image, label

    def _data_layer(self, num_threads=8, prefetch_buffer=100):
        with tf.variable_scope('data'):
            filenames = tf.placeholder(tf.string, shape=[None])
            # Load data from TFRecord files
            dataset = tf.data.TFRecordDataset(filenames)
            # dataset.map: Applies parse_tfrecord function to each element of this dataset,
            # and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
            dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=num_threads)
            # dataset.repeat(): Repeats this dataset 2 times
            dataset = dataset.repeat()
            # dataset.batch: Combines sequential elements of this dataset into batches.
            dataset = dataset.batch(self.batch_size)
            # dataset.prefetch: will start a background thread to populate a ordered buffer
            dataset = dataset.prefetch(prefetch_buffer)
            # make_initializable_iterator: returns the values from the fed Dataset of iterator.
            # Also, iterator doesn’t keep track of how many elements are present in the Dataset.
            # Hence, it is normal to keep running the iterator’s get_next operation till Tensorflow’s tf.errors.OutOfRangeError exception is occurred
            iterator = dataset.make_initializable_iterator()
        return iterator, filenames


    def _loss_functions(self, logits, labels):
        with tf.variable_scope('loss'):
             # target_prob tensor is constructed from the data labels
             # with a 1 in each of the elements corresponding to a label's value, and 0 everywhere else.
            target_prob = tf.one_hot(labels, self.num_classes)

            tf.losses.softmax_cross_entropy(target_prob, logits)
            # This function adds the given losses to the regularization losses.
            total_loss = tf.losses.get_total_loss()
        return total_loss

    def _optimizer(self, total_loss, global_step):
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            optimizer = optimizer.minimize(total_loss, global_step=global_step)
        return optimizer

    def _performance_metric(self, logits, labels):
        with tf.variable_scope("performance_metric"):
            # tf.argmax gives you the index of maximum value along the specified axis.
            predicted = tf.argmax(logits, axis=1)
            # Cast the labels to interger.
            labels = tf.cast(labels, tf.int64)
            # tf.equal() determines if the element in the first tensor equals the one in the second.
            # We get an array of bools (True and False).
            corrects = tf.equal(predicted, labels)
            # Cast the True and False values to 1 and 0.
            corrects = tf.cast(corrects, tf.float32)
            # tf.reduce_mean sums and averages all the values in the tensor
            accuracy = tf.reduce_mean(corrects)
        return accuracy

    def train(self):
        # Iteration number
        global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name='iter_number')

        # Training graph
        images, labels = self.iterator.get_next()
        images = tf.image.resize_bilinear(images, (self.image_height, self.image_width))
        # Logits is the output tensor of a classification network,
        # whose content is the unnormalized (not scaled between 0 and 1) probabilities.
        logits = model(images, num_classes=self.num_classes)
        loss = self._loss_functions(logits, labels)
        optimizer = self._optimizer(loss, global_step)
        accuracy = self._performance_metric(logits, labels)

        # Create a summary operation to log the progress of the network
        with tf.variable_scope('logging'):
            # summary placeholders
            streaming_loss_p = tf.placeholder(tf.float32)
            streaming_accuracy_p = tf.placeholder(tf.float32)

            summary_op_train = tf.summary.merge([
                tf.summary.scalar('streaming_loss', streaming_loss_p),
                tf.summary.scalar('streaming_accuracy', streaming_accuracy_p)
            ])
            summary_op_test = tf.summary.scalar('streaming_loss', streaming_loss_p)
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

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            # Resume training if a checkpoint exists
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(session, checkpoint.model_checkpoint_path)
                print('Loaded parameters from {}'.format(checkpoint.model_checkpoint_path))

            initial_step = global_step.eval()

            # Running training Set
            streaming_loss = 0
            streaming_accuracy = 0
            # One epoch is one full run through the training data set.
            for epoch in range(initial_step, self.training_epochs + 1):
                # Run the optimizer over and over to train the network.
                session.run(optimizer)
                # Get the current accuracy scores by running the "cost" operation on the training and test data sets
                loss_batch = session.run([loss, accuracy])

                streaming_loss += loss_batch
                streaming_accuracy += accuracy_batch
                if epoch % self.log_iter == self.log_iter - 1:
                    streaming_loss /= self.log_iter
                    print("Iteration: {}, Training loss: {:.2f}, Training accuracy: {:.2f}".format(epoch + 1, streaming_loss, streaming_accuracy))
                    training_summary = session.run(summary_op_train, feed_dict={streaming_loss_p: streaming_loss,
                                                                                streaming_accuracy_p: streaming_accuracy}})
                    # Write the current training status to the log files (Which we can view with TensorBoard)
                    training_writer.add_summary(training_summary, global_step=epoch)

                    streaming_loss = 0
                    streaming_accuracy = 0

                # Save model
                if epoch % self.save_iter == self.save_iter - 1:
                    save_path = saver.save(session, self.checkpoint_path, global_step=global_step)
                    print("Model saved!")

                # Run validation set
                if epoch % self.val_iter == self.val_iter - 1:
                    print("Running validation.")
                    session.run(self.iterator.initializer, feed_dict={self.filenames: self.validation_filenames})

                    validation_accuracy = 0
                    for j in range(self.num_val_samples // self.batch_size):
                        aaccuracy_batch = session.run(accuracy)
                        validation_accuracy += aaccuracy_batch
                    validation_accuracy /= j

                    print("Accuracy: {}".format(validation_accuracy))

                    testing_summary = session.run(summary_op_test, feed_dict={streaming_accuracy_p: validation_accuracy})
                    # Write the current testing status to the log files (Which we can view with TensorBoard)
                    testing_writer.add_summary(testing_summary, global_step=epoch)

                    # Switch back to the training set
                    session.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})

            training_writer.close()
            testing_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--data_path', type=str, default='./tfrecords/',
                        help="Path to the TFRecords")
    args = parser.parse_args()
    trainer = TFModelTrainer(args.checkpoint_path, args.data_path)
    trainer.train()

if __name__ == '__main__':
    main()
