import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MINST_data", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

corrent = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(corrent, "float"))
max_accuracy = 0

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    if i % 10 == 0:
        ac = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        max_accuracy = max(max_accuracy, ac)

print max_accuracy
sess.close()
