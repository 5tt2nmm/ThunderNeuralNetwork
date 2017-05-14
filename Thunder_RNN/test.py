import time
import cx_Oracle, datetime
import math, numpy as np,traceback
import csv
import collections
import operator
from PIL import Image, ImageDraw, ImageFont
import  setdata_RNN
import tensorflow as tf

batch_size=50

if __name__ == "__main__":
    setdata_RNN.getoracledata()


