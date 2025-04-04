import tensorflow as tf

class PEModel(tf.Module):
    def __init__(self):
        super(PEModel, self).__init__()
        self.conv1 = tf.Variable(tf.random.normal([3, 3, 1, 32]), trainable=True)
        self.conv2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), trainable=True)
        self.fc1_weights = None
        self.fc2_weights = tf.Variable(tf.random.normal([128, 1]), trainable=True)

    def __call__(self, x):
        x = tf.nn.conv2d(x, self.conv1, strides=1, padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = tf.nn.conv2d(x, self.conv2, strides=1, padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = tf.reshape(x, [x.shape[0], -1])

        if self.fc1_weights is None:
            self.fc1_weights = tf.Variable(tf.random.normal([x.shape[1], 128]), trainable=True)

        x = tf.matmul(x, self.fc1_weights)
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.fc2_weights)

        return tf.sigmoid(x)

    def save_weights(self, path):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(path)

    def load_weights(self, path):
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.restore(tf.train.latest_checkpoint(path)).expect_partial()
