# -*- coding: utf-8 -*-

import time
import cx_Oracle, datetime
import math, numpy as np,traceback
import csv
import collections
import operator
from PIL import Image, ImageDraw, ImageFont
import  setdata_RNN
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import tensorflow.contrib
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'



batch_size=50

if __name__ == "__main__":
    #setdata_RNN.getoracledata()
    x_batches,y_batches,datalen,n_chunk,datas_vector,data_num_map,datas=setdata_RNN.getbatchs()
    input_data = tf.placeholder(tf.int32, [None, None])
    output_targets = tf.placeholder(tf.int32, [None, None])


    # 定义RNN
    def neural_network(model='lstm', rnn_size=128, num_layers=2,keep_prob=0.5):
        if model == 'rnn':
            cell_fun = rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fun = rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fun = rnn_cell.BasicLSTMCell

        cell = cell_fun(rnn_size)
        cell=rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)
        cell = rnn_cell.MultiRNNCell([cell] * num_layers)

        initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, datalen + 1])
        softmax_b = tf.get_variable("softmax_b", [datalen + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [datalen + 1, rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)

        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
        output = tf.reshape(outputs, [-1, rnn_size])

        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        return logits, last_state, probs, cell, initial_state,inputs


    # 训练
    def train_neural_network():
        logits, last_state, probs, cell, initial_state,inputs = neural_network()
        targets = tf.reshape(output_targets, [-1])
        loss = seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],
                                                datalen)
        cost = tf.reduce_mean(loss)
        learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver(tf.all_variables())

            for epoch in range(50):
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
                n = 0
                for batche in range(n_chunk):
                    train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                    n += 1
                    print(epoch, batche, train_loss)
                    #print inputs.eval(feed_dict={input_data: x_batches[n]})




                if epoch % 7 == 0:
                    saver.save(sess, 'thundermodule', global_step=epoch)
                if epoch == 3:
                    # sess.run(tf.initialize_all_variables())

                    # saver = tf.train.Saver(tf.all_variables())
                    # saver.restore(sess, 'thundermodule-7')
                    state_ = sess.run(cell.zero_state(1, tf.float32))

                    # problabels=probs.eval(feed_dict={input_data: x_batches[0][1], initial_state: state_})
                    [probs_, state_] = sess.run([probs, last_state],
                                                feed_dict={input_data: np.array(x_batches[0][0]).reshape(1, 5),
                                                           initial_state: state_})
                    out = to_word(probs_, datas)
                    print out



    def to_word(predict, datas):
        # t = np.cumsum(predict[len(predict)-1])
        # s = np.sum(predict[len(predict)-1])
        # sample = int(np.searchsorted(t, np.random.rand(1) * s))

        idval=0
        for i in range(len(predict[4])):
            if predict[4][i]>idval:
                idval=predict[4][i]
                sample=i

        if sample > len(datas):
            sample = len(datas) - 1
        return datas[sample]



    train_neural_network()



