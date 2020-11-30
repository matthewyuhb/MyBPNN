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
#softmax��batch��ʽ��numpyʵ�ֻ�ͦ�鷳�ģ������Լ���̽��������Ҫ�ȶ�axis=1Ҳ���Ƕ�ÿһ�����Ȼ��ԭ����ת���ټ�����ת�û���
def softmax(x):
    #print("forward sofrmax")
    #�ҳ�ÿһ��(��Ӧһ������)�����ֵ������������Ԫ�ؼ�ȥ�����ֵ�õ���λ�������xȻ���ÿһ��λ����e��x����
    max_x = np.max(x,axis=1)
    x_T=x.T
    shift_x = (x_T-max_x).T
    exp_x = np.exp(shift_x)
    #�ҳ�ÿһ�еĺͣ�����������Ԫ�س��Ըú͵õ�softmax���ֵ
    sum_exp_x = np.sum(exp_x,axis=1)
    exp_xT=exp_x.T
    res=(exp_xT / sum_exp_x).T
    return res

def mse(target,output,batch_size=1):
    #print("forward mse")
    return 0.5 * np.sum( (target-output)**2 ) / batch_size


# ����mse�ĵ����Լ�cross_entropy_with_softmax�ĵ���һ�£��������ֻ��һ��loss���򴫲�������ƥ���������
# ��Ҫ���������������ֿ���������������+mse�Լ�softmax+cross_entropy
# ����ǰ�� ����Ȩ��ʱ����mse��bp�����������bp�����ں���ֱ����mse��bp����softmax+cross_entropy��bp
# ��ҲҪ��softmax���ʽ��������������֣�Ҳ���ܴ���������ʧ����
def loss_bp(target,output,batch_size=1):
    #print("start bp")
   # print("loss function bp")
    return (output-target) / batch_size # �˴���outputָ�����������output��������ʧ������ouotput

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