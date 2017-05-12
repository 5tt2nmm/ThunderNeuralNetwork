import os
import random
import re
import setdata_reCNN

import numpy as np
import tensorflow as tf

glosamplegroupcount1=1000
glosamplegroupcount2=1000
glosamplegroupcount3=1000
glosamplegroupcount4=200
glosampletotalcount=glosamplegroupcount1+glosamplegroupcount2+glosamplegroupcount3+glosamplegroupcount4
gloSampleTimecount=5
golsampleGridCount=21

StartTime='2016-07-26 12:00:00'


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')

if __name__ == "__main__":
    info = "./Thunder_reCNN/data_recnn"
    items = os.listdir(info)
    items.sort()
    Allsamples = np.zeros((1,golsampleGridCount*golsampleGridCount*gloSampleTimecount))
    Alllabels = np.zeros((1,4))
    for item in items:
        if "Samples" in item:
            print item
            tmpsample = np.loadtxt("Thunder_reCNN/data_recnn/" + item)
            Allsamples=np.append(tmpsample,Allsamples,axis=0)
            strinfo = re.compile('Samples')
            b = strinfo.sub('Labels', item)
            tmplabels = np.loadtxt("Thunder_reCNN/data_recnn/" + b)
            Alllabels = np.append(tmplabels,Alllabels,axis=0)

    Allsamples=Allsamples[1:]
    Alllabels=Alllabels[1:]
    thundersamples = Allsamples[glosamplegroupcount1:]
    thundersampleslabels = Alllabels[glosamplegroupcount1:]

    drop_samplecount = 0
    drop_traindata = []
    drop_trainlabels = []
    testdata=[]
    testlabels=[]
    trss = list(range(0, len(Allsamples)))
    trss_set = random.sample(trss, 2700)
    for j in range(len(Allsamples)):
        if j in trss_set:
            drop_traindata.append(Allsamples[j])
            drop_trainlabels.append(Alllabels[j])
        else:
            testdata.append(Allsamples[j])
            testlabels.append(Alllabels[j])

    # for all in range(0, len(Allsamples)):
    #     if Alllabels[all][0]!=1:
    #         thundersamples.append(Allsamples[all])
    #         thundersampleslabels.append(Alllabels[all])
    print "load all data"

    sess = tf.InteractiveSession()

    x = tf.placeholder("float", [None, 2205])
    y_ = tf.placeholder("float", [None, 4])

    # first convolutinal layer
    w_conv1 = weight_variable([5, 5, 5, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 21, 21, 5])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    w_fc1 = weight_variable([3 * 3 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    w_fc2 = weight_variable([1024, 4])
    b_fc2 = bias_variable([4])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    #cross_entropy =tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
    cross_entropy =tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv ,1), tf.argmax(y_ ,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())

    for i in range(20000):
        trss = list(range(0, len(drop_traindata)))
        trss_set = random.sample(trss, 100)
        tmpdata = []
        tmplabels =[]
        for s in range(len(trss_set)):
            tmpdata.append(drop_traindata[trss_set[s]])
            tmplabels.append(drop_trainlabels[trss_set[s]])
        feed_x = np.array(tmpdata)
        feed_y = np.array(tmplabels)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: testdata, y_: testlabels, keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
            print "LOSS:%f"%cross_entropy.eval(feed_dict={x: feed_x, y_: feed_y, keep_prob: 1.0})
            print "callback:%f" % accuracy.eval(feed_dict={x: thundersamples, y_: thundersampleslabels,keep_prob: 1.0})

            #print y_conv.eval(feed_dict={x: feed_x,y_: feed_y, keep_prob: 1.0})
            #print (y_ * tf.log(y_conv)).eval(feed_dict={x: feed_x, y_: feed_y, keep_prob: 1.0})


        train_step.run(feed_dict={x: feed_x, y_: feed_y, keep_prob: 0.7})
        if i==800:
            #print x_Grid.eval(feed_dict={x: thundersamples, keep_prob: 1.0})
            #print tf.nn.conv2d(x_Grid, W_conv1, strides=[1, 1, 1, 1], padding='SAME').eval(feed_dict={x: thundersamples, keep_prob: 1.0})
            numpy_list = np.asarray(y_conv.eval(feed_dict={x: testdata, keep_prob: 1.0}))
            np.savetxt("Thunder_reCNN/data_recnn/" + StartTime + "pred_labels.txt", numpy_list, fmt='%f')
            numpy_labellist = testlabels
            np.savetxt("Thunder_reCNN/data_recnn/" + StartTime + "sample_labels.txt", numpy_labellist, fmt='%f')