import math

def calculate(N):
    result = (4*(4*N)**1)*math.exp(-1/8*0.01*N)
    return result

a = [6000, 8000, 10000, 12000, 14000]

for i in a:
    print(calculate(i))