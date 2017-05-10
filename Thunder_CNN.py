import setdata_cnn
import random
import tensorflow as tf
import numpy as np
import os
import sys


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    StartTime='2016-05-02 00:00:00'

    #setdata_cnn.GetOracleDataSample(109.0, 25.0, '2016-04-02 00:00:00', '2016-04-07 00:00:00', 30, 5, 20, 0.5)
    #setdata_cnn.GetOracleDataSample(109.0, 25.0, '2016-07-29 00:00:00', '2016-08-02 00:00:00', 30, 5, 20, 0.5)
    #setdata_cnn.GetOracleDataSample(109.0, 25.0, '2016-06-01 00:00:00', '2016-06-04 00:00:00', 30, 5, 20, 0.5)
    #setdata_cnn.GetOracleDataSample(109.0, 25.0, '2016-07-13 00:00:00', '2016-07-19 00:00:00', 30, 5, 20, 0.5)
    #setdata_cnn.GetOracleDataSample(109.0, 25.0, '2016-08-04 00:00:00', '2016-08-08 00:00:00', 30, 5, 20, 0.5)

    info = "./data_cnn"
    items = os.listdir(info)
    Allsamples=np.zeros((1,2000))
    Alllabels=np.zeros((1,1))
    for item in items:
        if "SamplesArray" in item:
            print item
            tmpsample=np.loadtxt("data_cnn/"+item)
            ar1 = np.where(tmpsample > 200, 1, tmpsample)
            tmpsamplesconstant = np.where(ar1 > 0, ar1 / 200, ar1)
            # Allsamples.append(np.array(tmpsamplesconstant))
            Allsamples = np.concatenate((Allsamples, tmpsamplesconstant), axis=0)
            item.replace('SamplesArray', 'LabelsArray')

            tmplabels=np.loadtxt("data_cnn/"+item)
            ar1 = np.where(tmplabels > 200, 1, tmplabels)
            tmplabelsconstant = np.where(ar1 > 0, ar1 / 200, ar1)
            tmplabelsconstantarr = np.zeros((len(tmplabelsconstant), 1))
            for label in range(len(tmplabelsconstant)):
                tmplabelsconstantarr[label][0]=tmplabelsconstant[0][label]
            # Alllabels.append(np.array(tmplabelsconstant))
            Alllabels = np.concatenate((Alllabels, tmplabelsconstantarr), axis=0)

    # drop some empty sample
    drop_samplecount = 0
    drop_traindata = []
    drop_trainlabels = []
    for x in range(len(Alllabels)):
        if (Alllabels[x] > 0.0):
            drop_traindata.append(Allsamples[x])
            drop_trainlabels.append(Alllabels[x])
            drop_samplecount += 1
    for x2 in range(len(Alllabels)):
        if (Alllabels[x2] == 0.0):
            if drop_samplecount >=(len(Alllabels))/2:
                break
            drop_traindata.append(Allsamples[x2])
            drop_trainlabels.append(Alllabels[x2])
            drop_samplecount += 1


    #
    #
    # testSamplesfromtxt = np.loadtxt("data_cnn/2016-07-26 00:00:00SamplesArray")
    # testLabelsfromxt = np.loadtxt("data_cnn/2016-07-26 00:00:00LabelsArray")
    # ar1 = np.where(testSamplesfromtxt > 200, 1, testSamplesfromtxt)
    # testsamples = np.where(ar1 > 0, ar1 / 200, ar1)
    # ar2 = np.where(testLabelsfromxt > 200, 1, testLabelsfromxt)
    # testlabels = np.where(ar2 > 0, ar2 / 200, ar2)
    # print("testData empty Grid count:%s" % np.sum(testlabels == 0.0))
    #
    #
    # Samplesfromtxt = np.loadtxt("data_cnn/" + StartTime + "SamplesArray")
    # Labelsfromxt = np.loadtxt("data_cnn/" + StartTime + "LabelsArray")
    #
    # print np.max(Samplesfromtxt),np.max(Labelsfromxt)
    # print np.sum(Samplesfromtxt>200)
    #
    # ar1=np.where(Samplesfromtxt > 200,1,Samplesfromtxt)
    # samples=np.where(ar1 > 0, ar1/200,ar1)
    # ar2=np.where(Labelsfromxt > 200,1, Labelsfromxt)
    # labels=np.where(ar2 > 0, ar2/200, ar2)
    #
    # sslay = list(range(0, len(labels)))
    # trainset = random.sample(sslay, len(labels))
    #
    # traindata = []
    # trainlabels = []
    #
    #
    # for i in range( len(labels)):
    #     if i in trainset:
    #         traindata.append(samples[i])
    #         trainlabels.append(labels[i])
    #
    #
    #  # drop some empty sample
    # drop_samplecount = 0
    # drop_traindata = []
    # drop_trainlabels = []
    # for x in range(379):
    #     if (np.sum(trainlabels[x]>0.0)>0):
    #         drop_traindata.append(traindata[x])
    #         drop_trainlabels.append(trainlabels[x])
    #         drop_samplecount += 1
    # for x2 in range(379):
    #     if (np.sum(trainlabels[x2]>0.0)==0):
    #         if drop_samplecount == 300:
    #             break
    #         drop_traindata.append(traindata[x2])
    #         drop_trainlabels.append(trainlabels[x2])
    #         drop_samplecount += 1
    #
    # drop_traindata=np.concatenate((drop_traindata,testsamples),axis=0)
    # drop_trainlabels = np.concatenate((drop_trainlabels, testlabels), axis=0)

    thundersamples = []
    thundersampleslabels = []
    for all in range(len(drop_trainlabels)):
        if drop_trainlabels[all] >= 0.00001:
            thundersamples.append(drop_traindata[all])
            thundersampleslabels.append(drop_trainlabels[all])


    sess=tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, len(drop_traindata[0])])
    y_ = tf.placeholder(tf.float32, [None,  1])

    x_Grid=tf.reshape(x,[-1,20,20,5])
    #tg= x_Grid.eval(feed_dict={x: samples})
    #print tg[21][0][7]

    W_conv1 = weight_variable([3, 3, 5, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.sigmoid(conv2d(x_Grid, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # W_conv1 = weight_variable([1, 1, 5, 16])
    # b_conv1 = bias_variable([16])
    # h_conv1 = tf.nn.sigmoid(conv2d(x_Grid, W_conv1) + b_conv1)
    # # W_conv2 = weight_variable([3, 3, 8, 16])
    # # b_conv2 = bias_variable([16])
    # # h_conv2 = tf.nn.sigmoid(conv2d(h_conv1, W_conv2) + b_conv2)
    # W_fc1 = weight_variable([20 * 20 * 16, 1024])
    # b_fc1 = bias_variable([1024])
    # # h_pool2_flat = tf.reshape(h_conv2, [-1, 20 * 20 * 16])
    # h_fc1 = tf.nn.sigmoid(tf.matmul(tf.reshape(h_conv1,[-1, 20 * 20 * 16]), W_fc1) + b_fc1)
    # keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # W_fc2 = weight_variable([1024, 1])
    # b_fc2 = bias_variable([1])
    # y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # in_units = len(samples[0])
    # h1_units = 200
    # w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    # b1 = tf.Variable(tf.zeros([h1_units]))
    # w2 = tf.Variable(tf.zeros([h1_units, 1]))
    # b2 = tf.Variable(tf.zeros([1]))
    # x = tf.placeholder(tf.float32, [None, len(samples[0])])
    # keep_prob = tf.placeholder(tf.float32)
    # hidden1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    # hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    # y_conv = tf.nn.sigmoid(tf.matmul(hidden1_drop, w2) + b2)
    # y_ = tf.placeholder(tf.float32, [None, 1])
    # # loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))


    cross_entropy =5*tf.reduce_sum(tf.abs(y_conv-y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.round(y_conv*100), tf.round(y_*100))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        trss = list(range(0, len(thundersampleslabels)))
        trss_set = random.sample(trss, 10)
        tmpdata = []
        tmplabels =np.zeros( (10,1) )
        for j in range(len(trss_set)):
            tmpdata.append(thundersamples[trss_set[j]])
            tmplabels[j][0]=thundersampleslabels[trss_set[j]]
        feed_x = np.array(tmpdata)
        feed_y = np.array(tmplabels)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: Allsamples, y_: Alllabels.reshape(-1,1), keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
            print "LOSS:%f"%cross_entropy.eval(feed_dict={x: feed_x, y_: feed_y, keep_prob: 1.0})
            print "callback:%f" % accuracy.eval(feed_dict={x: thundersamples, y_: np.array(thundersampleslabels).reshape(len(thundersampleslabels), 1),                           keep_prob: 1.0})

        train_step.run(feed_dict={x: feed_x, y_: feed_y, keep_prob: 0.5})
        if i==200:
            numpy_list = np.asarray(y_conv.eval(feed_dict={x: thundersamples, keep_prob: 1.0}))
            np.savetxt("data_cnn/" + StartTime + "pred_labels.txt", numpy_list, fmt='%f')
            numpy_labellist = thundersampleslabels
            np.savetxt("data_cnn/" + StartTime + "sample_labels.txt", numpy_labellist, fmt='%f')

    print "test accuracy %g" % accuracy.eval(feed_dict={x: samples, y_:labels, keep_prob: 1.0})


