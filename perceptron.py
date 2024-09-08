import numpy as np

#and��or��nand��?�Ȫϫ�?����?����?�������ުǪ���
def and_gate(x1, x2):
    x = np.array(x1, x2)
    w = np.array([0.5 ,0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def not_and_gate(x1, x2):
    x = np.array(x1, x2)
    w = np.array([-0.5 ,-0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def or_gate(x1, x2):
    x = np.array(x1, x2)
    w = np.array([0.5 ,0.5])
    b = -0.2
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#������xor��?�Ȫ�perceptron���Ūê����ު���Ϊ���ʦ������
def xor_gate(x1, x2): 
    x = np.array(x1, x2)
    w = np.array([0.7 ,-0.4])
    b = -0.6
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
