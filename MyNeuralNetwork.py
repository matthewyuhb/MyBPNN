# -*- coding:utf-8 -*-
import pickle

import numpy as np
import random
import deeplearningtoolkit as dlt

###########################################################类代码开始####################################################
class myNeuralNetwork:
    # initialize the numbers of input layer、hidden layer、output layer's nodes not including the bias nodes!!!!
    # initialize the weights
    def __init__(self,layerNums,nodeNumsList,activationList,learningRate,epoch,loss='mse',batchSize = 1):
        # layerNums为3的三层网络权重层数为2
        # nodeNumsList有3个元素对应了3层包含的节点数
        # activationList有2个元素，对应了全连接后的激活方式（2=3-1）
        # loss代表了优化器的损失函数
        print(batchSize)
        self.layerNums = layerNums
        self.nodeNumsList=nodeNumsList
        self.activationList =[] #根据输入的字符串判别是否合法，合法将该字符串放入，否则放入‘sigmoid’
        self.activationFunctionsList=[] #激活函数列表（按顺序存储激活函数的指针）
        self.weightsList=[] #权重列表（存储fc层的权重）
        self.biasesList=[] #偏置列表（存储fc层的偏置）
        self.activationFunctionsbpList=[] #非线性激活函数反向传播的函数指针列表
        for i in range(self.layerNums-1):
            np.random.seed(1)
            self.weightsList.append(np.random.normal(0.0,pow(self.nodeNumsList[i],-0.5),(self.nodeNumsList[i],self.nodeNumsList[i+1])))
            self.biasesList.append(0.001*np.ones(self.nodeNumsList[i+1]))
            if activationList[i]=='sigmoid':
                self.activationList.append('sigmoid')
                self.activationFunctionsList.append(lambda x:dlt.sigmoid(x))
                self.activationFunctionsbpList.append(lambda dOutput,output:dlt.sigmoid_bp(dOutput,output))
            elif activationList[i]=='relu':
                self.activationList.append('relu')
                self.activationFunctionsList.append(lambda x:dlt.relu(x))
                self.activationFunctionsbpList.append(lambda dOutput,output:dlt.relu_bp(dOutput,output))
            elif activationList[i]=='tanh':
                self.activationList.append('tanh')
                self.activationFunctionsList.append(lambda x:dlt.tanh(x))
                self.activationFunctionsbpList.append(lambda dOutput,output:dlt.tanh_bp(dOutput,output))
            else:
                print("The activation function you choose is invalid,so we use sigmoid instead")
                self.activationList.append('sigmoid')
                self.activationFunctionsList.append(lambda x:dlt.sigmoid(x))
                self.activationFunctionsbpList.append(lambda dOutput,output:dlt.sigmoid_bp(dOutput,output))
        self.batchSize = batchSize
        self.lr = learningRate
        self.epoch = epoch
        self.loss = loss

        if self.loss == 'mse':
            self.lossFunction = lambda target,output,batchSize:dlt.mse(target,output,batchSize)
        else:
            print("The loss function you choose is invalid, so we use mse instead")
            self.lossFunction = lambda target,output,batchSize:dlt.mse(target,output,batchSize)

    def save(self):

        wfile = open('wfile.pickle', 'wb')
        pickle.dump(self.weightsList, wfile)
        wfile.close()
        bfile = open('bfile.pickle', 'wb')
        pickle.dump(self.biasesList, bfile)
        bfile.close()

    def load(self):
        self.weightsList=[]
        self.biasesList=[]

        wfile = open('wfile.pickle', 'rb')
        self.weightsList = pickle.load(wfile)
        bfile = open('bfile.pickle', 'rb')
        self.biasesList = pickle.load(bfile)



    #数据放进来训练前就asarray转换为array，别用list
    def train(self,train_samples,train_targets):
        loss=[]
        # self.load()

        for e in range(self.epoch):
            lr=self.lr#*(1.0/(1+0.001*e))
            dataList=[]
            sample_list=[i for i in range(len(train_samples))]
            sample_list=random.sample(sample_list,self.batchSize)
            batch_samples=[train_samples[i] for i in sample_list]
            batch_targets=[train_targets[i] for i in sample_list]

            dataList.append(np.array(batch_samples))
            output = batch_samples
            for i in range(self.layerNums-1):
                output = np.dot(output,self.weightsList[i])+self.biasesList[i]
                output = self.activationFunctionsList[i](output)
                dataList.append(output)
            loss.append(self.lossFunction(batch_targets,output,self.batchSize))
            #Error对Weights和biases的偏导数列表，再下面的循环是反着存的我们可以翻转过来再更新！
            dW=[]
            db=[]
            dOutput = dlt.loss_bp(batch_targets,output,self.batchSize)

            #按照计算机的下标从激活函数bp函数和weightsList的最后一层遍历到第0层，计算dW和db
            for i in range(self.layerNums-2,-1,-1):
                dOutput = self.activationFunctionsbpList[i](dOutput,dataList[i+1])
                ret = dlt.fc_bp(dOutput,output,dataList[i],self.weightsList[i],self.biasesList[i])
                #返回值ret为一个三元素的列表，第0个元素是继续向前传递的误差，第二个元素是dW，第三个是db
                dOutput = ret[0]
                dW.append(ret[1])
                db.append(ret[2])
            dW.reverse()
            db.reverse()
            #根据计算得到的dW和db以及学习率更新参数矩阵和偏置矩阵
            for i in range(self.layerNums-1):
                self.weightsList[i] -= lr* dW[i]
                self.biasesList[i] -= lr * db[i]
            print("epoch:"+str(e)+" loss:"+str(loss[e]))



        print(loss[10],loss[-10])
        return loss
    # test and show the results of input samples
    # input examples: [ [1 2]  [3 4]  [5 6]  ]
    def test(self,inputs):
        inputs = np.asarray(inputs)

        for i in range(self.layerNums-1):
            inputs = np.dot(inputs,self.weightsList[i])+self.biasesList[i]
            inputs = self.activationFunctionsList[i](inputs)
        return inputs




###########################################################类代码结束####################################################
