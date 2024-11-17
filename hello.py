w1 = 1
w2 = -1
b = 1

def xor_gate(x1, x2):
    if w1*x1 + w1*x2 - b <= 0:
        print(0)
    else:
        print(1)

xor_gate(0, 0)
xor_gate(1, 0)
xor_gate(0, 1)
xor_gate(1, 1)