# -*- coding: utf-8 -*-

import random
import tensorflow as tf
import numpy as np
def addLayer(inputData,inSize,outSize,activity_function = None):
	Weights = tf.Variable(tf.random_normal([inSize,outSize]))
	basis = tf.Variable(tf.zeros([1,outSize])+0.1)
	weights_plus_b = tf.matmul(inputData,Weights)+basis
	if activity_function is None:
		ans = weights_plus_b
	else:
		ans = activity_function(weights_plus_b)
	return ans
if __name__ == "__main__":

    #datamatrix,datalabels=SetData.GetOracleDataSample(108.3, 22.8, '2016-06-02 00:00:00', '2016-06-06 00:00:00', 6, 10, 800, 9, 0.1)
    datamatrix=np.load("Thunder_BP/datamatrix.npy")
    datalabels = np.load("Thunder_BP/datalabels.npy")
    print("Data empty Grid count:%s"%np.sum(datalabels==0.0))

    thundersamples=[]
    thundersampleslabels=[]
    for all in range(800):
        if datalabels[all]>=0.001:
            thundersamples.append(datamatrix[all])
            thundersampleslabels.append(datalabels[all])


    sslay = list(range(0, 800))
    trainset=random.sample(sslay, 600)

    traindata=[]
    trainlabels=[]
    testdata=[]
    testlabels=[]

    for i in range(800):
        if i in trainset:
            traindata.append(datamatrix[i])
            trainlabels.append(datalabels[i])
        else:
            testdata.append(datamatrix[i])
            testlabels.append(datalabels[i])

    #drop some empty sample
    drop_samplecount=0
    drop_traindata=[]
    drop_trainlabels=[]
    for x in range(600):
        if(trainlabels[x]>=0.001):
            drop_traindata.append(traindata[x])
            drop_trainlabels.append(trainlabels[x])
            drop_samplecount+=1
    for x2 in range(600):
        if(trainlabels[x2]<=0.001):
            if drop_samplecount==300:
                break
            drop_traindata.append(traindata[x2])
            drop_trainlabels.append(trainlabels[x2])
            drop_samplecount += 1


    inputtestdata=np.array(testdata)
    inputtestlabels=np.zeros((800-len(trainset),1))
    for s  in range(len(testlabels)):
        inputtestlabels[s][0]=testlabels[s]

    in_units=len(datamatrix[0])
    h1_units=200

    w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
    b1=tf.Variable(tf.zeros([h1_units]))
    w2=tf.Variable(tf.zeros([h1_units,1]))
    b2=tf.Variable(tf.zeros([1]))
    x=tf.placeholder(tf.float32, [None, len(datamatrix[0])])
    keep_prob=tf.placeholder(tf.float32)
    hidden1=tf.nn.sigmoid(tf.matmul(x,w1)+b1)
    hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
    y=tf.nn.sigmoid(tf.matmul(hidden1_drop,w2)+b2)
    y_=tf.placeholder(tf.float32,[None,1])
    #loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y*5-y_*5)))
    #train_step=tf.train.AdagradOptimizer(0.5).minimize(loss)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

    correct_prediction = tf.equal(tf.round(y*5), tf.round(y_*5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    #sess.run(tf.initialize_all_variables())
    for step in range(10000):
        trss = list(range(0, 300))
        trss_set = random.sample(trss, 100)
        tmpdata=[]
        tmplabels= np.zeros( (100,1) )
        for j in range(len(trss_set)):
            tmpdata.append(drop_traindata[trss_set[j]])
            tmplabels[j][0]=drop_trainlabels[trss_set[j]]
        feed_x= np.array(tmpdata)
        feed_y= np.array(tmplabels)


        if step % 500 == 0:
            print #y.eval(feed_dict={x: inputtestdata, keep_prob:1.0})
            print "LOSS:%f"%loss.eval(feed_dict={x: feed_x, y_: feed_y, keep_prob: 1.0})
            #print correct_prediction.eval(feed_dict={x:datamatrix, y_: datalabels.reshape(500,1), keep_prob: 1.0})
            #pre = tf.round(y* 20)
            #print pre.eval(feed_dict={x: inputtestdata, keep_prob:1.0})
            #labels = tf.round(y_* 20)
            #print labels.eval(feed_dict={y_: inputtestlabels})
            print "accuracy:%f"%accuracy.eval(feed_dict={x:datamatrix, y_: datalabels.reshape(800,1), keep_prob: 1.0})
            print "callback:%f"%accuracy.eval(feed_dict={x:thundersamples, y_: np.array(thundersampleslabels).reshape(len(thundersampleslabels),1), keep_prob: 1.0})
        if step == 4000:
            numpy_list = np.asarray(y.eval(feed_dict={x: inputtestdata, keep_prob: 1.0}))
            np.savetxt("./Thunder_BP/pred_labels.txt", (numpy_list),fmt='%f')
            numpy_labellist = np.asarray((inputtestlabels.reshape(200, 1)))
            np.savetxt("./Thunder_BP/sample_labels.txt", numpy_labellist,fmt='%f')


        train_step.run(feed_dict={x: feed_x, y_:feed_y,keep_prob: 0.8})


