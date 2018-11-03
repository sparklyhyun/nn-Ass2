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
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

no_epochs = 100  #originally 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

#is this a character model or word model???
def rnn_model(x):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=tf.nn.softmax)

    return logits, word_list

def data_read_words():
  
    x_train, y_train, x_test, y_test = [], [], [], []
  
    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))
  
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

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

    x_train, y_train, x_test, y_test, n_words = data_read_words()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    logits, word_list = rnn_model(x)

    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        acc = []
        for e in range(no_epochs):
            word_list_, _, loss_  = sess.run([word_list, train_op, entropy], {x: x_train, y_: y_train})
            loss.append(loss_)

            acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
            #acc.append(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
            #print('epoch: %d, accuracy: %g' %(e, acc[e]))

            if e%10 == 0:
                print('epoch: %d, entropy: %g'%(e, loss[e]))
                print('epoch: %d, accuracy: %g' %(e, acc[e]))

        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.savefig('figures/partb_4_entropy.png')

        pylab.figure(2)
        pylab.plot(range(len(acc)), acc)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.savefig('figures/partb_4_accuracy.png')

        pylab.show()
  
if __name__ == '__main__':
    main()
