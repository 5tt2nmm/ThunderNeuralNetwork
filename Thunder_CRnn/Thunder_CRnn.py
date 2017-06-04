# -*- coding: utf-8 -*-
import os
import random
import re

import numpy as np
import tensorflow as tf



glosamplegroupcount1=10000
glosamplegroupcount2=8000
glosamplegroupcount3=2000
glosamplegroupcount4=500
glosampletotalcount=glosamplegroupcount1+glosamplegroupcount2+glosamplegroupcount3+glosamplegroupcount4
gloSampleTimecount=5
golsampleGridCount=21

rnnsize=128
rnnlayers=2

learning_rate = tf.Variable(0.0, trainable=False)

batch_size = 50
sess = tf.InteractiveSession()

StartTime='2017-05-25 12:00:00'


def getVec(var):
    if var<0.0001:
        return 0
    elif 0.0499<=var and var <=0.1001:
        return 1
    elif 0.1001<var and var <0.4999:
        return 2
    else:
        return 3
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean=tf.reduce_mean(var)
        tf.summary.scalar("mean",mean)
        with tf.name_scope("stddev"):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev",stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_max(var))
        tf.summary.histogram("histogram",var)


input_data = tf.placeholder(tf.float32,[batch_size, 441,5])
output_targets = tf.placeholder(tf.int64, [None, None])
keep_prob = tf.placeholder(tf.float32)


def convpool_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights",
                              initializer=tf.truncated_normal(kernel_shape,stddev=0.1))
    variable_summaries(weights)
    # Create variable named "biases".
    biases = tf.get_variable("biases",
                             initializer=tf.constant(0.1, shape=bias_shape))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    variable_summaries(biases)
    h_conv= tf.nn.relu(conv + biases)
    h_pool=tf.nn.avg_pool(h_conv, ksize=[1, 3, 3, 1],
                    strides=[1, 3, 3, 1], padding='SAME')
    return  h_pool


def cnn_filter(input):
    x_image = tf.reshape(input, [-1, 21, 21, 1])
    with tf.name_scope("conv1"):
        with tf.variable_scope("conv1_var"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
            relu1 = convpool_relu(x_image, [5, 5, 1, 32], [32])
    with tf.name_scope("conv2"):
        with tf.variable_scope("conv2_var"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
            relu2= convpool_relu(relu1, [5, 5, 32, 64], [64])
    with tf.name_scope("fulcon"):
        with tf.variable_scope("fulcon_var"):
            w_fc1 = tf.get_variable("weights_fc",
                                  initializer=tf.truncated_normal([3 * 3 * 64, rnnsize],stddev=0.1))
            b_fc1 = tf.get_variable("biases_fc",
                                 initializer=tf.constant(0.1, shape=[rnnsize]))

    h_pool2_flat = tf.reshape(relu2, [-1, 3 * 3 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1_drop

CnnOut=[]
with tf.variable_scope("cnngrid_filters") as scope:
    result0 = cnn_filter(input_data[:,:,0])
    scope.reuse_variables()
    CnnOut.append(result0)
    result1 = cnn_filter(input_data[:, :, 1])
    scope.reuse_variables()
    CnnOut.append(result1)
    result2 = cnn_filter(input_data[:, :, 2])
    scope.reuse_variables()
    CnnOut.append(result2)
    result3 = cnn_filter(input_data[:, :, 3])
    scope.reuse_variables()
    CnnOut.append(result3)
    result4 = cnn_filter(input_data[:, :, 4])
    scope.reuse_variables()
    CnnOut.append(result4)



def lstm_cell():
    return  tf.contrib.rnn.LSTMCell(rnnsize,forget_bias=0.0,state_is_tuple=True)
def attn_cell(keep_prob):
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(),output_keep_prob=keep_prob)
cell=tf.contrib.rnn.MultiRNNCell([attn_cell(keep_prob) for _ in range(rnnlayers)],state_is_tuple=True)
_initial_state = cell.zero_state(batch_size, tf.float32)

outputs=[]
outputs_logs=[]
state=_initial_state
with tf.variable_scope("RNN"):
    input=CnnOut
    for time_step in range(gloSampleTimecount):
        if time_step>0:tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(input[time_step], state)
        outputs_logs.append(cell_output)
        outputs=cell_output

last_state=state

softmax_w = tf.get_variable("softmax_w",initializer=tf.truncated_normal([rnnsize, 4 ],stddev=0.1))
softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(0.1, shape=[4]))


output = tf.reshape(outputs, [-1, rnnsize])
outputs_log=tf.reshape(outputs_logs, [-1, rnnsize])

logits = tf.matmul(output, softmax_w) + softmax_b
log_logits= tf.matmul(outputs_log, softmax_w) + softmax_b



with tf.name_scope("result_pred"):
    probs = tf.nn.softmax(logits)
    tf.summary.histogram("probs",probs)
    cross_entropy =tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=output_targets))
    tf.summary.scalar("cross_entropy", cross_entropy)
#cross_entropy = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example([log_logits], [tf.reshape(output_targets,[-1])], [tf.ones([batch_size*gloSampleTimecount],dtype=tf.float32)]))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal( tf.argmax(output_targets, 1), tf.argmax(probs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)




merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("data_crnn/log/train", sess.graph)
test_writer=tf.summary.FileWriter("data_crnn/log/test")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.all_variables())


if __name__ == "__main__":

    info = "./data_crnn"
    items = os.listdir(info)
    items.sort()
    Allsamples = np.zeros((1, golsampleGridCount * golsampleGridCount * gloSampleTimecount))
    Alllabels = np.zeros((1, 4))
    for item in items:
        if "1Samples" in item:
            print item
            tmpsample = np.loadtxt("data_crnn/" + item)
            Allsamples = np.append(tmpsample, Allsamples, axis=0)
            strinfo = re.compile("1Samples")
            b = strinfo.sub('1Labels', item)
            tmplabels = np.loadtxt("data_crnn/" + b, dtype="Int32")
            Alllabels = np.append(tmplabels, Alllabels, axis=0)

    Allsamples = Allsamples[1:]
    Alllabels = Alllabels[1:]

    # #增加一个步骤，将样本标签制作成time——step的序列
    # Alllabels_vec=[]
    # for x in range (len(Allsamples)):
    #     v=Allsamples[x][(220*5+1):(220*5+5)]
    #     v_Vec=[getVec(v_tmp) for v_tmp in v]
    #     if Alllabels[x][0]==1:v_Vec.append(0)
    #     elif Alllabels[x][1]==1:v_Vec.append(1)
    #     elif  Alllabels[x][2] == 1:v_Vec.append(2)
    #     else: v_Vec.append(3)
    #     Alllabels_vec.append(v_Vec)
    #
    # Alllabels=np.array(Alllabels_vec)


    thundersamples = Allsamples[glosamplegroupcount1:]
    thundersampleslabels = Alllabels[glosamplegroupcount1:]

    drop_samplecount = 0
    drop_traindata = []
    drop_trainlabels = []
    testdata = []
    testlabels = []
    trss = list(range(0, len(Allsamples)))
    trss_set = random.sample(trss, 18000)
    for j in range(len(Allsamples)):
        if j in trss_set:
            drop_traindata.append(Allsamples[j])
            drop_trainlabels.append(Alllabels[j])
        else:
            testdata.append(Allsamples[j])
            testlabels.append(Alllabels[j])

    npdrop_traindata = np.array(drop_traindata).reshape([-1, 21 * 21, 5])
    nptestdata = np.array(testdata).reshape([-1, 21 * 21, 5])

    print "load data"


    for i in range(7500):

        size = len(nptestdata) / batch_size
        trss = list(range(0, len(npdrop_traindata)))
        trss_set = random.sample(trss, batch_size)
        #trss_set = list(range((i%size)*batch_size, ((i%size)+1)*batch_size))
        tmpdata = []
        tmplabels = []
        for s in range(len(trss_set)):
            tmpdata.append(npdrop_traindata[trss_set[s]])
            tmplabels.append(drop_trainlabels[trss_set[s]])
        feed_x = np.array(tmpdata)
        feed_y = np.array(tmplabels)
        if i % 100 == 0:
            state_ = sess.run(cell.zero_state(batch_size, tf.float32))


            to_ac=0
            for tmp in range(size):
                to_ac +=accuracy.eval(
                    feed_dict={input_data: nptestdata[(tmp*batch_size):((tmp+1)*batch_size)], output_targets: testlabels[(tmp*batch_size):((tmp+1)*batch_size)],
                               keep_prob: 1.0, _initial_state: state_})
            to_ac=to_ac/size


            print "LOSS:%f" % cross_entropy.eval(
                feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 1.0,_initial_state: state_})

            # print "input4"
            # print input_data[:, :, 4].eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y.reshape([batch_size,4]), keep_prob: 1.0, _initial_state: state_})
            #
            # print "CNNOUT4"
            # print CnnOut[4].eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y.reshape([batch_size,4]), keep_prob: 1.0, _initial_state: state_})

            # print "weights"
            #
            # with tf.variable_scope("cnngrid_filters/conv1", reuse=True):
            #     b1 = tf.get_variable("biases")
            # WB= sess.run([b1])
            # print WB
            # with tf.variable_scope("cnngrid_filters/conv2", reuse=True):
            #     b2 = tf.get_variable("biases")
            # WB = sess.run([b2])
            # print WB

            print "step %d, training accuracy %g" % (i, to_ac)

            print"sample & pres"
            # print output_targets.eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 1.0,_initial_state: state_})
            # print tf.argmax(probs, 1).eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 1.0,_initial_state: state_})


            # print y_conv.eval(feed_dict={x: feed_x,y_: feed_y, keep_prob: 1.0})
            # print (y_ * tf.log(y_conv)).eval(feed_dict={x: feed_x, y_: feed_y, keep_prob: 1.0})

        if  i% 500 == 0 and i >0:
            saver.save(sess, 'thundermodule', global_step=i)
        if i == 500:

            # print "output"
            # print output.eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 1.0, _initial_state: state_})
            # print "logits"
            # print logits.eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 1.0, _initial_state: state_})
            # print "probs"
            # print probs.eval(
            #     feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 1.0, _initial_state: state_})
            # print x_Grid.eval(feed_dict={x: thundersamples, keep_prob: 1.0})
            # print tf.nn.conv2d(x_Grid, W_conv1, strides=[1, 1, 1, 1], padding='SAME').eval(feed_dict={x: thundersamples, keep_prob: 1.0})
            numpy_list = np.asarray(probs.eval(feed_dict={input_data: nptestdata[0:batch_size], keep_prob: 1.0}))
            np.savetxt("data_crnn/" + StartTime + "pred_labels.txt", numpy_list, fmt='%f')
            numpy_labellist = testlabels[0:batch_size]
            np.savetxt("data_crnn/" + StartTime + "sample_labels.txt", numpy_labellist, fmt='%f')
        if i%10==0:

            summary,acc = sess.run([merged,accuracy], feed_dict={
                input_data: nptestdata[0:batch_size],
                output_targets: testlabels[0:batch_size], keep_prob: 1.0,
                _initial_state: state_})
            test_writer.add_summary(summary,i)

        if i%100==99:
            run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata=tf.RunMetadata()
            summary,_=sess.run([merged,train_step],feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 0.5,_initial_state: state_})
            train_writer.add_summary(summary,i)
        else:
            sess.run(tf.assign(learning_rate, 0.001 * (0.9999 ** i)))
            train_step.run(
                feed_dict={input_data: feed_x, output_targets: feed_y, keep_prob: 0.5, _initial_state: state_})
    train_writer.close()
    test_writer.close()









