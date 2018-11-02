import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab

import os

if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
batch_size = 128
embedding_size = 20

no_epochs = 10  #originally 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

#cnn model 
def word_cnn_model(x):
    #x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    print(x)

    embedding = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=embedding_size)
    print(embedding)

    #input_layer = tf.reshape((tf.one_hot(embedding, 256), tf.int64), [-1, MAX_DOCUMENT_LENGTH, 256, 1])
    input_layer = tf.reshape(tf.one_hot(tf.cast(embedding, tf.int64), 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    #convolutional & pooling layer 1
    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
            input_layer,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            activation=tf.nn.relu)
        print('before pool1')
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

        #pool1 = tf.squeeze(tf.reduce_max(pool1, 1), squeeze_dims=[1])
    #convolutional & pooling layer 2
    with tf.variable_scope('CNN_Layer2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters = N_FILTERS,
            kernel_size = FILTER_SHAPE2,
            padding='VALID',
            activation = tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size = POOLING_WINDOW,
            strides = POOLING_STRIDE,
            padding = 'SAME')

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])
    print('pool 2 done')
    #output softmax layer 
    logits = tf.layers.dense(pool2, MAX_LABEL, activation=tf.nn.softmax)

    return input_layer, logits

#data preprocessing 
def read_data_word():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words

def main():
    global n_words

    x_train, y_train, x_test, y_test, n_words = read_data_word()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    inputs, logits = word_cnn_model(x)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    #prediction part
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        loss_batch = []
        test_acc = []

        #breaking down into batches
        N = len(x_train)
        idx = np.arange(N)
        
        for e in range(no_epochs):
            np.random.shuffle(idx)
            trainX_batch, trainY_batch = x_train[idx], y_train[idx]

            #batch training
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                _, loss_  = sess.run([train_op, entropy], {x: trainX_batch[start:end], y_: trainY_batch[start:end]})
                loss_batch.append(loss_)

            loss.append(sum(loss_batch)/len(loss_batch))
            loss_batch[:] = []
            #test accuracy!!!!!!
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            #test_acc.append(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

            if e%1 == 0:
                print('iter: %d, entropy: %g'%(e, loss[e]))
                print('iter: %d, accuracy: %g'%(e, test_acc[e]))

        #plot figures here
        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.savefig('figures/partb_2_entropy.png')

        pylab.figure(2)
        pylab.plot(range(len(test_acc)), test_acc)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.savefig('figures/partb_2_accuracy.png')

        pylab.show()
        
        sess.close()

if __name__ == '__main__':
    main()

