
def f(s):
    a = 0;b = 0;c = 99;d = 99;
    ys = []
    for i in range(10):
        x = s[i]
        next_a = b
        next_b = -c+99
        next_c = d
        next_d = -x+99
        a = next_a;b = next_b;c = next_c;d = next_d;
        y = a
        ys.append(y)
    return ys
    