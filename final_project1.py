# -*- coding:utf-8 -*-
import MyNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt








############################################数据集制作####################################

fig=plt.figure()

ax3=plt.axes(projection='3d')

granularity=0.25#控制训练集的数量 以及 拟合的精细度
x1=np.arange(-5,5,granularity)
x2=np.arange(-5,5,granularity)
X1,X2=np.meshgrid(x1,x2)
Z=np.sin(X1)-np.cos(X2)

ax3.plot_surface(X1,X2,Z,alpha=0.7)

print(X1)
print(X2)
print(Z)


listcases=[]
listlabels=[]

for i in range(int(10/granularity)):
    for j in range(int(10/granularity)):
        listcases.append([X1[i][j],X2[i][j]])
        listlabels.append([(Z[i][j]+2.0)/4.0])
############################################训练####################################
listcases=np.array(listcases)
listlabels=np.array(listlabels)
# print(listcases)

ANN = MyNeuralNetwork.myNeuralNetwork(4,[2,40,40,1],['relu','relu','sigmoid'],0.8,5000,'mse',800)
ANN.train(listcases,listlabels)
ANN.save()
ANN.load()

listRes=ANN.test(listcases)
count=0
for i in range(int(10/granularity)):
    for j in range(int(10/granularity)):
        Z[i][j]=listRes[count]*4.0-2.0
        count+=1
print(listRes)


ax3.plot_surface(X1,X2,Z,cmap='rainbow')
                                                   
plt.show()

















































