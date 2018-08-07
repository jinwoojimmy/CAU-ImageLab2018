# 16/07/18 Daily Report

## Seminar

Undergraduate students and several graduate students in Image Lab gathered to hold a seminar.

Each undergraduate student prepared own section with selected paper chosen by individual preferences.

In addition, each student complemented last week's presentation by studying more on mis-understood part on last week. 

### CNN
Ki-Soo surveyed on CNN(Convolutional Neural Network). 

#### Intro
CNN is a kind of deep learning.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_inspiration.PNG" width="500" height="500">

The picture above indicates research by Hubel and Wiesel.

This shows observation of responding neuron when animal watches a certain object.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_inspiration2.PNG" width="500" height="500">

If we take a look at the result of observation, we can figure out that neurons are reacting on certain part of the object(not all of image).

Based on this inspiration, CNN was invented.

CNN shows good performance on various fields like image and text.

#### CNN experiment on MNIST
```python
    import tensorflow as tf
    import random

    from tensorflow.examples.tutorials.mnist import input_data

    tf.set_random_seed(777)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [None, 784])
    X_img = tf.reshape(X, [-1, 28, 28, 1]) # img 28 * 28 * 1
    Y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    '''
    Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
    Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
    Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
    '''

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    '''
    Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
    Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
    Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
    '''

    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

    '''
    Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
    Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
    Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
    Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
    Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
    '''

    W4 = tf.get_variable(name='9', shape=[128 * 4 * 4, 625],
                        initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([625]))
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    '''
    Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    '''

    W5 = tf.get_variable(name='0', shape=[625, 10],
                        initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L4, W5) + b5

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={ X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
```


And this is the result.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/cnn_seminar_result.PNG" width="200" height="400">


### Decaying LR vs Batch Normalization
Jin-Woo(Jeon) surveyed on decaying learning rate and batch normalization.

#### Implementation of MRF 
Presented on segmentation and denoising using MRF method.
  
