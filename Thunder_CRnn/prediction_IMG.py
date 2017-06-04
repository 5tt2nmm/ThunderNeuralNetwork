# -*- coding: utf-8 -*-
import os
import random
import re
import cx_Oracle,datetime

import numpy as np
import tensorflow as tf
import Thunder_CRnn
import math
import Image
import ImageDraw

modeldir="/home/leiyu/PycharmProjects/ThunderNeuralNetwork_master/Thunder_CRnn/"
batch_size=50
glosql = "select * from thunder2015"
glocountsql = "select count(*) from thunder2015"
connection = cx_Oracle.connect('thunder', 'thunder', '159.226.50.119:1521/thunder')


glostartLO=109.0
glostartLA=29.0
gloendLO=116.0
gloendLA=34.0

golstarttime='2017-05-11 06:36:00'
golendtime='2017-05-11 06:42:00'
T_gloStartTime=datetime.datetime.strptime(golstarttime, '%Y-%m-%d %H:%M:%S')
T_gloEndTime = datetime.datetime.strptime(golendtime, '%Y-%m-%d %H:%M:%S')
gloSampleTimecount=5
golsampleGridCount=21
golGridlen=0.1
glogridcountY=int((gloendLA-glostartLA)/golGridlen)
glogridcountX=int((gloendLO-glostartLO)/golGridlen)
golTimeStep=6

proimgX=8180
proimgY=350

YSIZE=6713
def lolatoimgpoint(lo,la):
    try:
        varpi=math.pi/180
        x= lo * 20037508.34 / 180.0
        y = math.log(math.tan((90 + la) * varpi / 2.0)) / varpi
        y = y * 20037508.34 / 180.0

        imgx=int(x/ 1000.0) - proimgX
        imgy = YSIZE-(int(y/ 1000.0) - proimgY)
        if imgy<3035:
            print lo,la

        return imgx,imgy
    except Exception,e:
        print lo,la


def getsample(lo,la,time):
    sample=np.zeros((golsampleGridCount*golsampleGridCount*gloSampleTimecount))
    s_start_lo=lo-((golsampleGridCount-1)/2)*golGridlen
    s_end_lo=s_start_lo+golsampleGridCount*golGridlen
    s_start_la = la - ((golsampleGridCount - 1) / 2) * golGridlen
    s_end_la = s_start_la + golsampleGridCount * golGridlen
    s_start_time=time-datetime.timedelta(minutes=golTimeStep*gloSampleTimecount)
    cursor = connection.cursor()
    sql = glosql + " where longitude>=" + str(s_start_lo) + " and longitude<" + str(s_end_lo) + " and latitude>=" + str(
        s_start_la) + " and latitude<" + str(
        s_end_la) +" and datetime>= to_date('" + datetime.datetime.strftime(s_start_time,'%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')" + " and datetime< to_date('" +  datetime.datetime.strftime(time,'%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')"
    #print(sql)
    cursor.execute(sql)
    result = cursor.fetchall()
    for line in result:
        tmpLO = line[1]
        tmpLA = line[0]
        tmpdatetime = line[6]
        X = int((tmpLO - s_start_lo) / (golGridlen))
        Y = int((tmpLA - s_start_la) / (golGridlen))
        divtime = tmpdatetime - time
        timeidx = int((divtime.total_seconds()) / (golTimeStep * 60))
        sample[(Y * golsampleGridCount + X)*gloSampleTimecount+timeidx] += 1
    cursor.close()
    for it in range(len(sample)):
        if sample[it]>=20:
            sample[it]=1.0
        else:
            sample[it]=sample[it]/20.0
    return sample

def getall():

    testAll=[]
    PredAll=[]
    for grididx in range(glogridcountX*glogridcountY):
        if grididx%1000==0:
            print grididx
        xidx = grididx % glogridcountX
        yidx = grididx / glogridcountX
        tmpstartLO = glostartLO + xidx * golGridlen
        tmpstartLA = glostartLA + yidx * golGridlen
        tmpendLO = tmpstartLO + golGridlen
        tmpendLA = tmpstartLA + golGridlen
        cursor = connection.cursor()
        sql = glocountsql + " where longitude>=" + str(tmpstartLO) + " and longitude<" + str(
            tmpendLO) + " and latitude>=" + str(
            tmpstartLA) + " and latitude<" + str(
            tmpendLA) + " and datetime>= to_date('" + datetime.datetime.strftime(T_gloStartTime,
                                                                                 '%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')" + " and datetime< to_date('" + datetime.datetime.strftime(
            T_gloEndTime, '%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')"
        # print(sql)
        cursor.execute(sql)
        result = cursor.fetchall()
        line = result[0]
        tmpcount = line[0]
        if tmpcount == 0:
            tmpcount = 0
        elif tmpcount >= 1 and tmpcount <= 2:
            tmpcount = 1
        elif tmpcount > 2 and tmpcount < 10:
            tmpcount = 2
        elif tmpcount > 10:
            tmpcount = 3
        wstartX,wstartY=lolatoimgpoint(tmpstartLO,tmpstartLA)
        wendX, wendY = lolatoimgpoint(tmpendLO, tmpendLA)
        testAll.append(list([wstartX, wstartY, wendX, wendY, tmpcount]))
        sample = getsample(tmpstartLO, tmpstartLA, T_gloStartTime)
        PredAll.append(sample)
    np.savetxt("data_crnn/testAll", np.array(testAll), fmt='%f')
    np.savetxt("data_crnn/PredAll", np.array(PredAll), fmt='%f')
    print "getall data"
if __name__ == "__main__":

    #getall()
    predsample = np.loadtxt("data_crnn/PredAll")
    predsample=predsample.reshape([-1,21*21,5])
    testAll = np.loadtxt("data_crnn/testAll")



    #
    # input_datas = tf.placeholder(tf.float32, [batch_size, 441,5])
    #
    # keep_prob = tf.placeholder(tf.float32)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(modeldir)
        saver.restore(sess, checkpoint)

        size=(glogridcountX*glogridcountY)/batch_size

        AllPredData = np.zeros(((glogridcountX*glogridcountY),5))

        for step in range(size):
            state_ = sess.run(Thunder_CRnn.cell.zero_state(batch_size, tf.float32))
            [predict]= sess.run([Thunder_CRnn.probs],
                                             feed_dict={Thunder_CRnn.input_data: np.array(predsample[step*batch_size:(step+1)*batch_size]),Thunder_CRnn.keep_prob: 1.0})
            preddata=sess.run(tf.argmax(predict, 1))
            for tmp in range(batch_size):
                s=step*batch_size+tmp
                la=int(s/glogridcountX)*golGridlen+glostartLA
                lo = int(s % glogridcountX)*golGridlen+glostartLO
                endla=la+golGridlen
                endlo=lo+golGridlen
                wstartX, wstartY = lolatoimgpoint(lo, la)
                wendX, wendY = lolatoimgpoint(endlo, endla)
                AllPredData[s]=np.array(list([wstartX, wstartY, wendX, wendY, preddata[tmp]]))

        pre_acc=sess.run(tf.reduce_mean(tf.cast(tf.equal(AllPredData[:,4],testAll[:,4]), "float")))
    print "acc is %g" %pre_acc

    img= Image.open('ProMD.jpg')
    drawObject = ImageDraw.Draw(img)

    BudstartX,BudstartY=lolatoimgpoint(glostartLO,glostartLA)
    BudendX, BudendY = lolatoimgpoint(gloendLO, gloendLA)
    drawObject.rectangle([(BudstartX, BudendY), (BudendX, BudstartY)],outline = "black")

    for tmpdata in testAll:
        tmpBudstartX, tmpBudstartY = tmpdata[0], tmpdata[1]
        tmpBudendX, tmpBudendY = tmpdata[2], tmpdata[3]
        if tmpdata[4]==1:
            drawObject.rectangle([(tmpBudstartX, tmpBudendY), (tmpBudendX, tmpBudstartY)], fill = (0, 255, 0, 1))
        elif tmpdata[4]==2:
            drawObject.rectangle([(tmpBudstartX, tmpBudendY), (tmpBudendX, tmpBudstartY)], fill = (0, 0, 255, 1))
        elif tmpdata[4]==3:
            drawObject.rectangle([(tmpBudstartX, tmpBudendY), (tmpBudendX, tmpBudstartY)], fill = (204, 0, 0, 1))




    img.save('sample_ProMD.jpg')
    del img

    img = Image.open('ProMD.jpg')
    drawObject = ImageDraw.Draw(img)

    BudstartX, BudstartY = lolatoimgpoint(glostartLO, glostartLA)
    BudendX, BudendY = lolatoimgpoint(gloendLO, gloendLA)
    drawObject.rectangle([(BudstartX, BudendY), (BudendX, BudstartY)], outline="black")

    for tmppredata in AllPredData:
        tmpBudstartX, tmpBudstartY = tmppredata[0], tmppredata[1]
        tmpBudendX, tmpBudendY = tmppredata[2], tmppredata[3]
        if tmppredata[4] == 1:
            drawObject.rectangle([(tmpBudstartX, tmpBudendY), (tmpBudendX, tmpBudstartY)], fill=(0, 255, 0, 1))
        elif tmppredata[4] == 2:
            drawObject.rectangle([(tmpBudstartX, tmpBudendY), (tmpBudendX, tmpBudstartY)], fill=(0, 0, 255, 1))
        elif tmppredata[4] == 3:
            drawObject.rectangle([(tmpBudstartX, tmpBudendY), (tmpBudendX, tmpBudstartY)], fill=(204, 0, 0, 1))

    img.save('prediction_ProMD.jpg')

