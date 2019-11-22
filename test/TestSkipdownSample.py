import numpy as np
input = np.arange(0, 32000)
print(len(input))
desiredN = 320
print(desiredN)
skip = len(input)//desiredN
print("skip:", skip)
inputNewnew = range(0, len(input), skip)
print(inputNewnew)


for i in range(len(input)):
    print(input[i])

print("")

for i in range(len(inputNewnew)):
    print(inputNewnew[i])
print()
print(input[-1])
print(inputNewnew[-1])
print()
print(len(input))
print(len(inputNewnew))