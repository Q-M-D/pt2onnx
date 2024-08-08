import numpy as np
import os, sys


def make_data():
    n1, n2 = np.random.randint(1, 10000), np.random.randint(1, 10000)
    return n1, n2, n1 + n2

if __name__ == "__main__":
    np.random.seed(0)
    data = []
    for i in range(10000):
        n1, n2, n3 = make_data()
        data.append([n1, n2, n3])
    # save data
    data = np.array(data)
    if os.path.exists(os.path.dirname(__file__)+"/data") == False:
        os.makedirs(os.path.dirname(__file__)+"/data")
    np.save(os.path.dirname(__file__)+"/data/data.npy", data)