def and_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 1.0
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1