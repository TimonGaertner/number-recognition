from itertools import count

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
showgraph = False
y=[]
index = count()
def animate(i):

    plt.cla()
    plt.plot(y)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy in %")









def main():
    plt.style.use("fivethirtyeight")

    plt.figure(figsize=(15, 7))
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.show()


def append2y(num):
    y.append(num)