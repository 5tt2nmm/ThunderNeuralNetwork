# -*- coding: utf-8 -*-
import time
import cx_Oracle, datetime
import math, numpy as np,traceback
import csv
import collections
import operator
from PIL import Image, ImageDraw, ImageFont


import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'



glosql = "select latitude,longitude  from thunder2015"
glocountsql = "select count(*) from thunder2015"
connection = cx_Oracle.connect('********', '*********', '************:*****/****')
glosamplegroupcount=5000


glostartLO=108.3
glostartLA=22.8
gloendLO=110.3
gloendLA=24.8

batch_size =50

golstarttime='2016-07-26 12:00:00'
golendtime='2016-08-15 12:00:00'
T_gloStartTime=datetime.datetime.strptime(golstarttime, '%Y-%m-%d %H:%M:%S')
T_gloEndTime = datetime.datetime.strptime(golendtime, '%Y-%m-%d %H:%M:%S')
glonum_step=5
golsampleGridCount=20
golGridlen=0.1
glogridcount=int((gloendLA-glostartLA)/golGridlen)
golTimeStep=6


def getoracledata():
    ListAllGrid={}
    tmptime = T_gloStartTime
    ##先计算所有格网的时间数据
    timeidx=0
    while (tmptime < T_gloEndTime):
        tmptimeend = tmptime + datetime.timedelta(minutes=golTimeStep)
        sql = glosql + " where longitude>=" + str(glostartLO) + " and longitude<" + str(
            gloendLO) + " and latitude>=" + str(
            glostartLA) + " and latitude<" + str(
            gloendLA) + " and datetime>= to_date('" + datetime.datetime.strftime(tmptime, '%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')" + " and datetime< to_date('" + datetime.datetime.strftime(tmptimeend, '%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')"
        cursor = connection.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        tmplist =  np.zeros((golsampleGridCount*golsampleGridCount))
        tmplistlo,tmplistla=[],[]
        for line in result:
            tmpLO = line[1]
            tmpLA = line[0]
            tmplistlo.append(tmpLO)
            tmplistla.append(tmpLA)
            X = int((tmpLO - glostartLO) / (golGridlen))
            Y = int((tmpLA - glostartLA) / (golGridlen))
            tmplist[Y * golsampleGridCount + X]+=1
        cursor.close()
        wLO,wLA,wY,wX=0,0,0,0
        if tmplistlo.__len__()>0 and tmplistla.__len__()>0:
            wLO = sum(tmplistlo) / (len(tmplistlo))
            wLA = sum(tmplistla) / (len(tmplistla))
            wY = int((wLO - glostartLO) / (golGridlen * 5))
            wX = int((wLA - glostartLA) / (golGridlen * 5))
        wpoint=wX*(golsampleGridCount/5)+wY
        ListAllGrid[timeidx]={'list':tmplist,'wpoint':wpoint}
        tmptime = tmptime + datetime.timedelta(minutes=golTimeStep)
        timeidx+=1
        if timeidx%100==0:
            print timeidx
    print "getalldatagrid"
    # 对格网数据进行整理
    ListAllData = []
    for i in range(len(ListAllGrid)):
        if  sum(ListAllGrid[i]['list'])<5:
            totalcount=0
        elif sum(ListAllGrid[i]['list']) < 50:
            totalcount = 1
        elif sum(ListAllGrid[i]['list']) < 300:
            totalcount = 2
        else:
            totalcount = 3

        if  max(ListAllGrid[i]['list'])<3:
            maxsinglecount=0
        elif max(ListAllGrid[i]['list']) < 5:
            maxsinglecount = 1
        elif max(ListAllGrid[i]['list']) < 10:
            maxsinglecount = 2
        else:
            maxsinglecount = 3

        if   np.sum(np.array(ListAllGrid[i]['list']) > 0)<5:
            girdwiththundercount=0
        elif  np.sum(np.array(ListAllGrid[i]['list']) > 0) < 100:
            girdwiththundercount = 1
        elif  np.sum(np.array(ListAllGrid[i]['list']) > 0) < 200:
            girdwiththundercount = 2
        else:
            girdwiththundercount = 3

        point = ListAllGrid[i]['wpoint']
        tmp=list()
        tmp.append(totalcount)
        tmp.append(maxsinglecount)
        tmp.append(girdwiththundercount)
        tmp.append(point)
        ListAllData.append(tmp)
    np.savetxt("data_rnn/Samples", np.array(ListAllData), fmt='%d')

def getbatchs():
    #对格网数据进行整理
    ListData=np.loadtxt("data_rnn/Samples")
    counter = collections.Counter(tuple(map(tuple, ListData)))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    datas, _ = zip(*count_pairs)
    data_num_map = dict(zip(datas, range(len(datas))))
    to_num = lambda data: data_num_map.get(data, len(datas))
    datas_vector = list(map(to_num, tuple(map(tuple, ListData))))

    n_chunk = (len(datas_vector)-10)// batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        xdata = np.zeros((batch_size,glonum_step))
        ydata = np.zeros((batch_size, glonum_step))
        for j in range(batch_size):
            xdata[j]=datas_vector[start_index+j:start_index+glonum_step+j]
            ydata[j] = datas_vector[start_index + j+1:start_index + glonum_step + j+1]

        x_batches.append(xdata)
        y_batches.append(ydata)
    return  x_batches,y_batches,len(datas),n_chunk,datas_vector,data_num_map,datas