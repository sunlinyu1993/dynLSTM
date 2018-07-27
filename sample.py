# import tensorflow as tf
#
# from numpy.random import RandomState
#
# batch_size=8
#
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#
# x= tf.placeholder(tf.float32, shape=(None,2), name='x_input')
# y1= tf.placeholder(tf.float32, shape=(None,1), name='y_input')
#
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
#
# y=tf.sigmoid(y)
#
# cross_entropy=-tf.reduce_mean(y1*tf.log(tf.clip_by_value(y, 1**(-10), 1.0))+(1-y1)*tf.log(tf.clip_by_value(y, 1**(-10), 1.0)))
# train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#
# rdm = RandomState(1)
# dataset_size=128
# X= rdm.rand(dataset_size,2)
# Y=[[int(x1+x2<1)] for (x1,x2) in X]
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(w1))
#     print(sess.run(w2))
#
#     STEPS=5000
#     for i in range(STEPS):
#         start= (i*batch_size)%dataset_size
#         end=min(start+batch_size,dataset_size)
#         sess.run(train_step,feed_dict={x:X[start:end],y1:Y[start:end]})
#         if i % 1000==0:
#             total_cross_enropy=sess.run(cross_entropy,feed_dict={x:X,y1:Y})
#             print(total_cross_enropy)

# import tensorflow as tf
# a = tf.constant([1 , 2 , 3])
# b = tf.constant([4 , 5 , 6])
#
# c = tf.stack([a , b] , axis=0)
# d = tf.unstack(c , axis=0)
# e = tf.unstack(c , axis=1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(c))
#     print(sess.run(d))
import numpy as np
from keras.layers import Lambda,Input
# from keras.models import *
# from keras.layers import *
# model=Sequential()
# model.add(Embedding(input_dim=1000,output_dim=64,input_length=10))
# input_array = np.random.randint(low=1000, size=(32, 10))
# input_array2=np.random.random(size=[2,3])
# print(input_array2)
#
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)
# assert output_array.shape == (32, 10, 64)
# import tensorflow as tf
# import numpy as np
#
# a=np.random.random(size=[2,3])
# b=np.random.random(size=[2,1])
# tf.expand_dims()
# from keras import backend as K
# from keras.engine.topology import Layer
# import tensorflow as tf
#
# class MyMeanPool(Layer):
#     def __init__(self, axis, **kwargs):
#         self.supports_masking = True
#         self.axis = axis
#         super(MyMeanPool, self).__init__(**kwargs)
#
#     def compute_mask(self, input, input_mask=None):
#         # need not to pass the mask to next layers
#         return None
#
#     def call(self, x, mask=None):
#         if mask is not None:
#             mask = K.repeat(mask, x.shape[-1])
#             mask = tf.transpose(mask, [0,2,1])
#             mask = K.cast(mask, K.floatx())
#             x = x * mask
#             return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
#         else:
#             return K.mean(x, axis=self.axis)
#
#     def compute_output_shape(self, input_shape):
#         output_shape = []
#         for i in range(len(input_shape)):
#             if i!=self.axis:
#                 output_shape.append(input_shape[i])
#         return tuple(output_shape)
#
# from keras.layers import Input, Masking
# from keras.models import Model
#
#
# data = [[[10,10],[0, 0 ],[0, 0 ],[0, 0 ]],
#         [[10,10],[20,20],[0, 0 ],[0, 0 ]],
#         [[10,10],[20,20],[30,30],[0, 0 ]],
#         [[10,10],[20,20],[30,30],[40,40]]]
#
# A = Input(shape=[None,4,2]) # None * 4 * 2
# mA = Masking()(A)
# out = MyMeanPool(axis=1)(mA)
#
# model = Model(inputs=[A], outputs=[out])
#
# print (model.summary())
# print (model.predict(data))
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import tensorflow as tf
from keras.models import Model

# mu = np.array([[1, 5]])
# sampleNo=1000
# Sigma = np.array([[1, 0.5], [1.5, 3]])
#
# R = cholesky(Sigma)
#
# s = np.dot(np.random.randn(sampleNo, 2), R) + mu
#
# plt.subplot(144)
#
# # 注意绘制的是散点图，而不是直方图
#
# plt.plot(s[:,0],s[:,1],'+')
#
# plt.show()
# p=tf.random_normal([2,1000])
# x=np.array([[1,0.5],[1.5,3]])
# b=tf.reduce_mean(x,axis=1)
# sess=tf.Session()
#
# c=sess.run(b)
# j=sess.run(p)
# plt.scatter(j)
# print(j)
# print(c)
cont_limit=600

# a=np.random.randint(0, 10000, (300, cont_limit))
# j=tf.cast(a,dtype=tf.bool)
# k=tf.expand_dims(tf.reduce_sum(tf.cast(j, tf.int32), axis=1), axis=1)
# sess=tf.Session()
# print(a)
# n=sess.run(j)
# print(n)
# l=sess.run(k)
# print(l)
# a=tf.random_normal(shape=[2,3,4,5])
# q=tf.shape(a)[:-1]
# u=tf.concat([q, [8, -1]], 0)
# # j=a.get_shape().dims
# # u=j[-1]
# # newshape=j[:-1]+10+u//10
#
# sess=tf.Session()
# i=sess.run(u)
# print(i)
a=-1e12
print(a)