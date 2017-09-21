import tensorflow as tf

# get input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/mnist_data/", one_hot=True)

# setup the variables for the input
x = tf.placeholder(tf.float32, [None, 784]) 

# placeholder for "true" values
y_ = tf.placeholder(tf.float32, [None, 10])

# setup the layers of the fully connected neural net
hidden1 = tf.layers.dense(x, 300, name="hidden1", activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 100, name="hidden2", activation=tf.nn.relu)
y = tf.layers.dense(hidden2, 10, name="outputs")

# calculate the cross-entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# setup the training step
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# some tensorflow boilerplate to initialize everything
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train the model for 1000 steps
for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# test the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
