# -*- coding: utf-8 -*-
import time
import cx_Oracle, datetime
import math, numpy as np,random
import csv
from PIL import Image, ImageDraw, ImageFont
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

connection = cx_Oracle.connect('*****', '*****', '*******/****')

#read_oracle

def GetOracleDataSample(CenterLO, CenterLA, StartTime, EndTime, TimeStep,SampleTimecount, SampleCOunt, GridCount, Gridlen):

    for tt in range(0, 5):
        print tt


    CenterGridStartlo = CenterLO - 0.5 * Gridlen
    CenterGridEndlo = CenterLO + 0.5 * Gridlen
    CenterGridStartla = CenterLA - 0.5 * Gridlen
    CenterGridEndla = CenterLO + 0.5 * Gridlen
    GridStartla = CenterLA - 0.5 * Gridlen - int(GridCount / 2) * Gridlen
    GridStartl0 = CenterLO - 0.5 * Gridlen - int(GridCount / 2) * Gridlen

    centergrididx=int(GridCount*GridCount/2)

    T_StartTime=datetime.datetime.strptime(StartTime, '%Y-%m-%d %H:%M:%S')
    T_EndTime = datetime.datetime.strptime(EndTime, '%Y-%m-%d %H:%M:%S')
    divtime=T_EndTime - T_StartTime
    totalTimestep = int((divtime.total_seconds()) / (TimeStep *60))

    tmptimelist = list(range(10, totalTimestep))
    SampleTimeIndexs = random.sample(tmptimelist, SampleCOunt)
    ListAllGridData = {}
    samplematrix = [[0.0 for i in range(SampleTimecount*2*GridCount*GridCount)] for i in range(SampleCOunt)]
    samplelabels=[0.0 for i in range(SampleCOunt)]

    for i in range(0, GridCount * GridCount):
        yidx = int(i / GridCount)
        xidx = int(i % GridCount)
        tmpstartLO=GridStartl0+xidx*Gridlen
        tmpstartLA=GridStartla+yidx*Gridlen
        tmpendLO=tmpstartLO+Gridlen
        tmpendLA=tmpstartLA+Gridlen
        cursor = connection.cursor()
        sql = r"select t.*,s.SUMINTENS from(select  to_char(datetime,'yyyy-mm-dd hh24')||':'||LPAD(floor(minute/6)*6,2,'0')||':00'  time, count(*) count from thunder2015 where dateTime>to_date('" + StartTime + "','yyyy-mm-dd hh24:mi:ss') and dateTime<to_date('" + EndTime + "','yyyy-mm-dd hh24:mi:ss') and longitude>=" + str(tmpstartLO) + " and longitude<" + str(tmpendLO) + " and latitude>=" + str(tmpstartLA) + " and latitude<" + str(tmpendLA) + " group by to_char(datetime, 'yyyy-mm-dd hh24')||':'||LPAD(floor(minute/6)*6,2,'0')||':00' order by  to_char(datetime, 'yyyy-mm-dd hh24')||':'||LPAD(floor(minute/6)*6,2,'0')||':00') t left join (select  to_char(datetime, 'yyyy-mm-dd hh24')||':'||LPAD(floor(minute/6)*6,2,'0')||':00'  time, sum(abs(INTENS)) SUMINTENS from thunder2015 where dateTime>to_date('" + StartTime + "','yyyy-mm-dd hh24:mi:ss') and dateTime<to_date('" + EndTime + "','yyyy-mm-dd hh24:mi:ss') and longitude>=" + str(tmpstartLO) + " and longitude<" + str(tmpendLO) + " and latitude>=" + str( tmpstartLA) + " and latitude<" + str(tmpendLA) + " group by to_char(datetime, 'yyyy-mm-dd hh24')||':'||LPAD(floor(minute/6)*6,2,'0')||':00' order by  to_char(datetime, 'yyyy-mm-dd hh24')||':'||LPAD(floor(minute/6)*6,2,'0')||':00') s on t.time=s.time"
        sql.strip().lstrip().rstrip(',')
        cursor.execute(sql)
        result = cursor.fetchall()
        GridInfos = {}
        for line in result:
            tmptime=line[0]
            tmpcount=line[1]
            tmpsumintens=line[2]
            GridInfos[tmptime]={'count':float(tmpcount),'sumintens':tmpsumintens}
        ListAllGridData[i]=GridInfos
    for j in range(0,len(SampleTimeIndexs)):
        tmpstarttime=T_StartTime+datetime.timedelta(minutes=SampleTimeIndexs[j]*TimeStep)
        if datetime.datetime.strftime(tmpstarttime, '%Y-%m-%d %H:%M:%S') in ListAllGridData[centergrididx].keys():
            if ListAllGridData[centergrididx][datetime.datetime.strftime(tmpstarttime, '%Y-%m-%d %H:%M:%S')]['count']>= 20:
                samplelabels[j]=1
            else:
                samplelabels[j]=ListAllGridData[centergrididx][datetime.datetime.strftime(tmpstarttime, '%Y-%m-%d %H:%M:%S')]['count']/20
        for grididx in range(0,(GridCount*GridCount)):
            for timediv in range(0,SampleTimecount):
                tmpsamplestarttime=tmpstarttime-datetime.timedelta(minutes=(timediv+1)*TimeStep)
                if datetime.datetime.strftime(tmpsamplestarttime, '%Y-%m-%d %H:%M:%S') in ListAllGridData[grididx].keys():

                    samplematrix[j][grididx*SampleTimecount*2+timediv*2]=ListAllGridData[grididx][datetime.datetime.strftime(tmpsamplestarttime, '%Y-%m-%d %H:%M:%S')]['count']
                    samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2+1] =  ListAllGridData[grididx][datetime.datetime.strftime(tmpsamplestarttime, '%Y-%m-%d %H:%M:%S')]['sumintens']
                    if samplematrix[j][grididx*SampleTimecount*2+timediv*2] >= 20:
                        samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2] = 1.0
                    else:
                        samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2] = \
                            samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2]/ 20
                    if samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2+1]  >= 2000:
                        samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2 + 1] = 1.0
                    else:
                        samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2 + 1] = \
                            samplematrix[j][grididx * SampleTimecount * 2 + timediv * 2 + 1] / 2000
    numpy_list = np.asarray(samplematrix)
    np.save("./datamatrix", numpy_list)
    numpy_labellist = np.asarray(samplelabels)
    numpy_labellist.reshape(SampleCOunt,1)
    np.save("./datalabels", numpy_labellist)
    return samplematrix,samplelabels

