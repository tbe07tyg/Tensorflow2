


lr = 1e-6
decay = 0.1
for i in range(1000):
    step =  i+1
    lr = lr *( 1. / (1. + decay * step))
    if lr < 3e-5:
        print(i,"less than 3e-5")

    print("%s: %s"% (step, lr))