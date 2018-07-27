import tensorflow as tf
import os
import numpy as np
import keras
from sklearn.utils import shuffle

np.random.seed(53)
epochs=24
numclass=4
totalnum=5
num_utterance=0
mask_value=100#遮蔽的数值
batch_size=256
squlength=50
datadim=32
accsegmentall=[]
accallutter=[]
accplus=0
accsegplus=0
proall=[]
n_hidden=64


def Traversingotherfiles(traverspath,total):
    finallotherpath=[]
    traindata=[]
    path = traverspath
    for index, peoplefiles in enumerate(os.listdir(path)):
        if index != total*2 and index !=total*2+1:
            emotionpath = os.path.join(path, peoplefiles)
            for emotionfiles in os.listdir(emotionpath):
                csvfilespath = os.path.join(emotionpath, emotionfiles)
                for csvfiles in os.listdir(csvfilespath):
                    finallpath = os.path.join(csvfilespath, csvfiles)
                    fa = open(finallpath)
                    x_temptraindata = []
                    for line in fa:
                        xstrain = line.split(',')
                        xstrain = list(map(lambda x: float(x), xstrain))
                        x_temptraindata.append(xstrain)
                    x_traindata = np.array(x_temptraindata)
                    traindata.append(x_traindata)
    return traindata

#遍历一个文件夹
def Traversingfiles(traverspath, total):
    testdata = []
    path = traverspath
    for index, peoplefiles in enumerate(os.listdir(path)):
        if index == total*2 or index== total*2+1:
            emotionpath = os.path.join(path, peoplefiles)
            for emotionfiles in os.listdir(emotionpath):
                csvfilespath = os.path.join(emotionpath, emotionfiles)
                for csvfiles in os.listdir(csvfilespath):
                    finallpath = os.path.join(csvfilespath, csvfiles)
                    fb = open(finallpath)
                    x_temptestdata = []
                    for line in fb:
                        xstest = line.split(',')
                        xstest = list(map(lambda x: float(x), xstest))
                        x_temptestdata.append(xstest)
                    x_datatest = np.array(x_temptestdata)
                    testdata.append(x_datatest)
    return testdata

#遍历验证集
# def Traversingvailfiles(traverspath, total):
#     vaildata = []
#     path = traverspath
#     for index, peoplefiles in enumerate(os.listdir(path)):
#         if index == total*2+1:
#             emotionpath = os.path.join(path, peoplefiles)
#             for emotionfiles in os.listdir(emotionpath):
#                 csvfilespath = os.path.join(emotionpath, emotionfiles)
#                 for csvfiles in os.listdir(csvfilespath):
#                     finallpath = os.path.join(csvfilespath, csvfiles)
#                     fb = open(finallpath)
#                     x_tempvaildata = []
#                     for line in fb:
#                         xsvail = line.split(',')
#                         xsvail = list(map(lambda x: float(x), xsvail))
#                         x_tempvaildata.append(xsvail)
#                     x_datavail = np.array(x_tempvaildata)
#                     vaildata.append(x_datavail)
#     return vaildata

def get_random_block_from_data(x_data,y_data,batch_size,i,numclass):
    x_data_batch=[]
    y_data_batch=[]
    samplenum=x_data.shape[0]
    if (i+1)*batch_size<=samplenum:
        x_data_batch=x_data[i * batch_size:(i + 1) * batch_size, :, :]
        y_data_batch=y_data[i*batch_size:(i+1)*batch_size,:]
        # y_data_batch=np.reshape(y_data_batch,[-1,numclass])#?
    if (i+1)*batch_size>samplenum:
        x_data_batch=x_data[i*batch_size:samplenum,:,:]
        y_data_batch=y_data[i*batch_size:samplenum,:]
        # y_data_batch = np.reshape(y_data_batch, [-1, numclass])#?
    return x_data_batch,y_data_batch
# def dynamicLSTM(x, y, difference_length,label):
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
#     outputs,_=tf.nn.dynamic_rnn(lstm_cell,x,sequence_length=difference_length,dtype=tf.float32)
#     output=outputs[:,-1,:]
#     predictions =tf.contrib.layers.fully_connected(inputs=output,num_outputs=4,activation=tf.nn.softmax)
#     loss=tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=predictions)
#     return predictions

# loss=dynamicLSTM(x,y,difference_length)
def RNN(X, weights, biases):
    X=tf.reshape(X,[-1,datadim])
    X_in= tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in,[-1,squlength,n_hidden])
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden)
    #两个状态c和h初始化为0
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    #dynamic_rnn 接收张量(batch_size,steps,inputs)
    outputs, final_state=tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_in, initial_state=init_state, time_major=False)
    results=tf.matmul(final_state[1],weights['out'])+biases['out']
    return results
# 分割数据
def split_data_label(data):
    dataall=np.concatenate(data[0:],axis=0)
    data_only=dataall[:,0:-1]
    data_label=dataall[:,-1]
    return data_only,data_label



#读取数据
total=1
traindata = Traversingotherfiles('F:/DataSet/校对过的数据集/only_hap/09emotion/IEMOCAP_09emo_mat_32_csv_seq_50', total=total)
testdata = Traversingfiles('F:/DataSet/校对过的数据集/only_hap/09emotion/IEMOCAP_09emo_mat_32_csv_seq_50',total=total)
# vaildata = Traversingvailfiles('F:/DataSet/校对过的数据集/only_hap/09emotion/IEMOCAP_09emo_mat_32_csv',total=total)
x_train, y_train = split_data_label(traindata)
x_train = np.reshape(x_train, [-1, squlength,datadim])
y_train=y_train[::squlength]
y_train = keras.utils.to_categorical(y_train, numclass)
x_train_shuffle,y_train_shuffle=shuffle(x_train,y_train)
#处理验证集
# x_vail,y_vail=split_data_label(vaildata)
# x_vail = np.reshape(x_vail, [-1, squlength, datadim])
# y_vail = y_vail[::squlength]
# y_vail = keras.utils.to_categorical(y_vail, numclass)
#处理测试集
x_test, y_test= split_data_label(testdata)
x_test=np.reshape(x_test,[-1,squlength,datadim])
y_test=y_test[::squlength]
y_test=keras.utils.to_categorical(y_test,numclass)
x_test_shuffle,y_test_shuffle=shuffle(x_test,y_test)
#定义权重
weights={
    'in': tf.Variable(tf.random_normal([datadim,n_hidden])),
    'out':tf.Variable(tf.random_normal([n_hidden,numclass]))
}
biases={
    'in': tf.Variable(tf.constant(0.1,shape=[n_hidden,])),
    'out':tf.Variable(tf.constant(0.1,shape=[numclass,]))
}
#设置输入变量
inputs = tf.placeholder(tf.float32,[None,squlength,datadim])
labels = tf.placeholder(tf.float32,[None,numclass])
#输入到RNN中
pred=RNN(inputs,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=labels))
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(labels,1))
acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
totalsteps=int(x_train_shuffle.shape[0]/batch_size)
totalteststeps=int(x_test_shuffle.shape[0]/batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):#每一轮epoch


        for j in range(totalsteps):

            x_train_batch, y_train_batch = get_random_block_from_data(x_train_shuffle, y_train_shuffle, batch_size, j, numclass)
            sess.run(train_op, feed_dict={inputs: x_train_batch, labels: y_train_batch})

            # acc,cost=sess.run([acc,cost],feed_dict={inputs: x_train_batch, labels: y_train_batch})
            print("Step " + str(j) + ", Minibatch Loss= " + \
 \
                  str(sess.run(cost, feed_dict={inputs: x_train_batch, labels: y_train_batch})) + ", Training Accuracy= " + \
 \
                  str(sess.run(acc,feed_dict={inputs: x_train_batch, labels: y_train_batch})))
    print("Optimization Finished!")

    correct_pre_out_all = []
    for v in range(totalteststeps):
        x_test_batch, y_test_batch = get_random_block_from_data(x_test_shuffle, y_test_shuffle, batch_size, v,
                                                                numclass)

        correct_pre_out = sess.run(correct_pred, feed_dict={inputs: x_test_batch, labels: y_test_batch})

        correct_pre_out_all.append(correct_pre_out)

    correct_pre_out_all_con = np.concatenate((correct_pre_out_all), axis=0)
    acc = tf.reduce_mean(tf.cast(correct_pre_out_all_con, tf.float32))
    WA=sess.run(acc)
    print(i)
    print("Testing Accuracy:"+str(WA))