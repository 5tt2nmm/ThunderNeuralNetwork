# -*- coding: utf-8 -*-
import time
import cx_Oracle, datetime
import math, numpy as np,traceback
import csv
from PIL import Image, ImageDraw, ImageFont

glosql = "select * from thunder2015"
connection = cx_Oracle.connect('****', '***', '**************:*****/*****')


def GetOracleDataSample(StartLO, StartLA, StartTime, EndTime, TimeStep,SampleTimecount, GridCount, Gridlen):



    T_StartTime = datetime.datetime.strptime(StartTime, '%Y-%m-%d %H:%M:%S')
    T_EndTime = datetime.datetime.strptime(EndTime, '%Y-%m-%d %H:%M:%S')
    divtime = T_EndTime - T_StartTime
    totalTimestep = int((divtime.total_seconds()) / (TimeStep * 60))
    DataArray=np.zeros((totalTimestep,GridCount*GridCount))


    # 建立游标
    print "start"
    cursor = connection.cursor()
    endLO = StartLO + GridCount * Gridlen
    endLA = StartLA + GridCount * Gridlen
    sql = glosql + " where longitude>=" + str(StartLO) + " and longitude<" + str(endLO) + " and latitude>=" + str(
        StartLA) + " and latitude<" + str(endLA) + " and datetime>= to_date('" +  StartTime + "','yyyy-mm-dd hh24:mi:ss')" + " and datetime< to_date('" + EndTime + "','yyyy-mm-dd hh24:mi:ss')"
    print(sql)
    cursor.execute(sql)
    result = cursor.fetchall()
    for line in result:
        tmpLO = line[1]
        tmpLA = line[0]
        tmpdatetime=line[6]
        X = int((tmpLO - StartLO) / (Gridlen))
        Y = int((tmpLA-StartLA  ) / (Gridlen))
        divtime=tmpdatetime-T_StartTime
        timeidx=int((divtime.total_seconds()) / (TimeStep * 60))
        DataArray[timeidx][Y*GridCount+X] += 1
    print "calculate DATA SUCCESS"
    np.savetxt("data_cnn/"+StartTime+"dataArray", DataArray,fmt='%f')
    # cursor.close
    # connection.close()

    SampleArray=np.zeros(((totalTimestep-SampleTimecount),GridCount*GridCount*SampleTimecount))
    labelsArray=np.zeros(((totalTimestep-SampleTimecount),1))
    for s in range(0,totalTimestep-SampleTimecount):
        #print s
        #tmpArr1=DataArray[s+SampleTimecount]
        #tmpArr2=tmpArr1.reshape(GridCount,GridCount)
        #tmpArr3=tmpArr2[(GridCount/2-5):(GridCount/2+5),(GridCount/2-5):(GridCount/2+5)]
        #tmpArr3 = tmpArr2[8:12][8:12]
        labelsArray[s]=DataArray[s+SampleTimecount][GridCount*(GridCount/2)+GridCount/2]
        for k in range(0,GridCount*GridCount):
           for m in range(0,SampleTimecount):
               SampleArray[s][k*SampleTimecount+m]=DataArray[s+m][k]


    # 写一个txt保存结果#
    np.savetxt("data_cnn/" + StartTime + "SamplesArray", SampleArray,fmt='%f')
    np.savetxt("data_cnn/" + StartTime + "LabelsArray", labelsArray,fmt='%f')
    print "DONE,data:"+StartTime

    return

