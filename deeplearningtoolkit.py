import numpy as np
def sigmoid(x):
    #print("forward sigmoid")
    return 1/(1+np.exp(-x))

def sigmoid_bp(dOutput,output):
    #print("bp sigmoid")
    return dOutput * output * (1.0 - output)

def relu(x):
    #print("forward relu")
    return np.maximum(0,x)

def relu_bp(dOutput,output):
    #print("bp relu")
    dOutput[output<=1e-7] = 0
    return dOutput

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_bp(dOutput,output):
    return dOutput*(1-output*output)
#softmax的batch形式用numpy实现还挺麻烦的，不如自己编程解决，否则要先对axis=1也就是对每一行求和然后将原矩阵转置再减和再转置回来
def softmax(x):
    #print("forward sofrmax")
    #找出每一行(对应一条样本)的最大值，将各行所有元素减去该最大值得到移位后的输入x然后对每一个位置求e的x次幂
    max_x = np.max(x,axis=1)
    x_T=x.T
    shift_x = (x_T-max_x).T
    exp_x = np.exp(shift_x)
    #找出每一行的和，将各行所有元素除以该和得到softmax输出值
    sum_exp_x = np.sum(exp_x,axis=1)
    exp_xT=exp_x.T
    res=(exp_xT / sum_exp_x).T
    return res

def mse(target,output,batch_size=1):
    #print("forward mse")
    return 0.5 * np.sum( (target-output)**2 ) / batch_size


# 由于mse的导数以及cross_entropy_with_softmax的导数一致，因此我们只用一种loss反向传播函数来匹配两种情况
# 需要在神经网络类中区分开两种情况：激活函数+mse以及softmax+cross_entropy
# 对于前者 更新权重时先做mse的bp再做激活函数的bp，对于后者直接做mse的bp代替softmax+cross_entropy的bp
# 这也要求softmax激活方式不能再隐含层出现，也不能搭配其他损失函数
def loss_bp(target,output,batch_size=1):
    #print("start bp")
   # print("loss function bp")
    return (output-target) / batch_size # 此处的output指的是神经网络的output而不是损失函数的ouotput

def cross_entropy(target,output,batch_size=1):
  #  print("forward:cross_entropy")
    return -np.sum(target*np.log(output+1e-7)) / batch_size

def identical_bp(dOutput,output):
 #   print("sofrmax bp")
    return dOutput

def fc_bp(dOutput,output,x,W,b):
#    print("fc bp")
    ret = []
    dInput = np.dot(dOutput,W.T)
    # print("W"+str(W)+"type of W:"+str(type(W)))
    # print("x"+str(x)+"type of x:"+str(type(x)))
    ret.append(dInput)
    dW = np.dot(x.T,dOutput)
    ret.append(dW)
    db = np.sum(dOutput,axis=0)
    ret.append(db)
    return ret