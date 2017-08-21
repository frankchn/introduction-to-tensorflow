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
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# calculate and output accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create summaries for tensorboard
tf.summary.scalar("cross_entropy", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
merged = tf.summary.merge_all()

# some tensorflow boilerplate to initialize everything
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
summary_writer = tf.summary.FileWriter("/tmp/mnist_logs", sess.graph)

# train the model for 1000 steps and write summaries
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    summary_writer.add_summary(summary, i)

# test the model
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
