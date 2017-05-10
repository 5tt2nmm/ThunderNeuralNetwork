import numpy as np
if __name__ == "__main__":
    StartTime = '2016-07-26 00:00:00'
    # setdata_cnn.GetOracleDataSample(109.0, 25.0, '2016-07-26 00:00:00', '2016-07-31 00:00:00', 30, 5, 20, 0.5)
    predictlabels = np.loadtxt("data_cnn/" + StartTime + "pred_labels.txt")
    samplelabels = np.loadtxt("data_cnn/" + StartTime + "sample_labels.txt")
    prear1 = np.where(True,np.round(predictlabels*20) , 0)
    samplearr = np.where(True, np.round(samplelabels * 20), 0)
    np.savetxt("data_cnn/" + StartTime + "round_pred_labels", prear1, fmt='%d')
    np.savetxt("data_cnn/" + StartTime + "round_sample_labels", samplearr, fmt='%d')