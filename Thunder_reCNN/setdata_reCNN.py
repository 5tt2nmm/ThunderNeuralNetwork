import time
import cx_Oracle, datetime
import math, numpy as np,traceback
import csv
from PIL import Image, ImageDraw, ImageFont


glosql = "select * from thunder2015"
glocountsql = "select count(*) from thunder2015"
connection = cx_Oracle.connect('thunder', 'thunder', '159.226.50.119:1521/thunder')
glosamplegroupcount1=1000
glosamplegroupcount2=1000
glosamplegroupcount3=1000
glosamplegroupcount4=200


glostartLO=109.0
glostartLA=25.0
gloendLO=119.0
gloendLA=35.0

golstarttime='2016-07-26 12:00:00'
golendtime='2016-07-26 12:06:00'
T_gloStartTime=datetime.datetime.strptime(golstarttime, '%Y-%m-%d %H:%M:%S')
T_gloEndTime = datetime.datetime.strptime(golendtime, '%Y-%m-%d %H:%M:%S')
gloSampleTimecount=5
golsampleGridCount=21
golGridlen=0.1
glogridcount=int((gloendLA-glostartLA)/golGridlen)
golTimeStep=6



samples1,labels1,samples2,labels2,samples3,labels3,samples4,labels4=[],[],[],[],[],[],[],[]


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

def getAll():
    glo_zeroeffnum = 500
    glo_zerowithdata = 0
    tmptime=T_gloStartTime
    while (tmptime<T_gloEndTime):
        tmptimeend=tmptime+datetime.timedelta(minutes=golTimeStep)
        for grididx in range(glogridcount*glogridcount):
            xidx=grididx/glogridcount
            yidx=grididx%glogridcount
            tmpstartLO = glostartLO + xidx * golGridlen
            tmpstartLA = glostartLA + yidx * golGridlen
            tmpendLO = tmpstartLO + golGridlen
            tmpendLA = tmpstartLA + golGridlen
            cursor = connection.cursor()
            sql = glocountsql + " where longitude>=" + str(tmpstartLO) + " and longitude<" + str(
                tmpendLO) + " and latitude>=" + str(
                tmpstartLA) + " and latitude<" + str(
                tmpendLA) + " and datetime>= to_date('" + datetime.datetime.strftime(tmptime,'%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')" + " and datetime< to_date('" +  datetime.datetime.strftime(tmptimeend,'%Y-%m-%d %H:%M:%S') + "','yyyy-mm-dd hh24:mi:ss')"
            #print(sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            line=result[0]
            tmpcount=line[0]
            if tmpcount ==0:
                if len(samples1)<glosamplegroupcount1:
                    #tmpsample=getsample(tmpstartLO, tmpstartLA, tmptime)
                    tmplabels = [1, 0, 0, 0]
                    tmpsample = getsample(tmpstartLO, tmpstartLA, tmptime)
                    samples1.append(tmpsample)
                    labels1.append(tmplabels)
                elif glo_zerowithdata<glo_zeroeffnum:
                    tmplabels = [1, 0, 0, 0]
                    tmpsample = getsample(tmpstartLO, tmpstartLA, tmptime)
                    npar = np.array(tmpsample)
                    if (np.sum(npar>0))>0:
                        #samples1.remove(0)
                        samples1[glo_zerowithdata]=tmpsample
                        #labels1.remove(0)
                        labels1[glo_zerowithdata]=tmplabels
                        glo_zerowithdata = glo_zerowithdata + 1

            elif tmpcount>=1 and tmpcount<=2:
                if len(samples2) < glosamplegroupcount2:
                    tmpsample =getsample(tmpstartLO, tmpstartLA, tmptime)
                    tmplabels = [0, 1, 0, 0]
                    samples2.append(tmpsample)
                    labels2.append(tmplabels)
                    if (len(samples1) >=glosamplegroupcount1 and len(samples2)  >=glosamplegroupcount2 and  len(samples3) >=glosamplegroupcount3 and  + len(samples4)  >=glosamplegroupcount4 ):
                        return

            elif tmpcount>2 and tmpcount<10:
                if len(samples3) < glosamplegroupcount3:
                    tmpsample =getsample(tmpstartLO, tmpstartLA, tmptime)
                    tmplabels = [0, 0, 1, 0]
                    samples3.append(tmpsample)
                    labels3.append(tmplabels)
                    if (len(samples1) >= glosamplegroupcount1 and len(samples2) >= glosamplegroupcount2 and len(
                            samples3) >= glosamplegroupcount3 and + len(samples4) >= glosamplegroupcount4):
                        return

            elif tmpcount>10:
                if len(samples4) < glosamplegroupcount4:
                    tmpsample =getsample(tmpstartLO, tmpstartLA, tmptime)
                    tmplabels = [0, 0, 0, 1]
                    samples4.append(tmpsample)
                    labels4.append(tmplabels)
                    if (len(samples1) >= glosamplegroupcount1 and len(samples2) >= glosamplegroupcount2 and len(
                            samples3) >= glosamplegroupcount3 and + len(samples4) >= glosamplegroupcount4):
                        return

        print len(samples1) , len(samples2) , len(samples3) , len(samples4)
        tmptime=tmptime+datetime.timedelta(minutes=golTimeStep)


if __name__ == "__main__":
    getAll()
    np.savetxt("Thunder_reCNN/data_recnn/1Samples1", np.array(samples1), fmt='%f')
    np.savetxt("Thunder_reCNN/data_recnn/1Labels1",  np.array(labels1), fmt='%d')

    np.savetxt("Thunder_reCNN/data_recnn/1Samples2", np.array(samples2), fmt='%f')
    np.savetxt("Thunder_reCNN/data_recnn/1Labels2", np.array(labels2), fmt='%d')

    np.savetxt("Thunder_reCNN/data_recnn/1Samples3", np.array(samples3), fmt='%f')
    np.savetxt("Thunder_reCNN/data_recnn/1Labels3", np.array(labels3), fmt='%d')

    np.savetxt("Thunder_reCNN/data_recnn/1Samples4", np.array(samples4), fmt='%f')
    np.savetxt("Thunder_reCNN/data_recnn/1Labels4", np.array(labels4), fmt='%d')

    print "DONE,data:"
