import tensorflow as tf
import numpy as np
data=np.zeros((50,50))

for i in range(0,50):
    data[i]=range(i*50,i*50+50)

num_steps=5
batch_size=1
sess = tf.InteractiveSession()

i=0

s=tf.placeholder("float", [None, 50])

x = tf.slice( s, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
x.set_shape([batch_size, num_steps])
y = tf.slice( s, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
y.set_shape([batch_size, num_steps+1])

with sess.as_default():
    print x.eval(feed_dict={s: np.array(data)})
    print y.eval(feed_dict={s: np.array(data)})


