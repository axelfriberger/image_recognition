

s = """import matplotlib.pyplot as plt
import numpy as np

s1='1 0.209...'
s2='1 0.2117...'
s3='1 0.210550'

s4='1 0.3400'
s5='1 0.305'
s6='1 0.304'

def create_lists(s, time=False):
    l = s.split()
    if time:
        x_axis = [int(i) for i in l if l.index(i) % 3 == 0]
        acc_axis = [round(float(i), 3) for i in l if l.index(i) % 3 == 1]
        time_axis = [round(float(i), 3) for i in l if l.index(i) % 3 == 2]
        return x_axis, acc_axis, time_axis

    else:
        x_axis = [int(i) for i in l if l.index(i) % 2 == 0]
        y_axis = [round(float(i), 3) for i in l if l.index(i) % 2 == 1]
        return x_axis, y_axis


def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text= "Max acc at: x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, verticalalignement="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


def find_list(l1, l2, l3, e):
    for i in l1:
        if i==e:
            return l1
    for i in l2:
        if i==e:
            return l2
    for i in l3:
        if i==e:
            return l3


plt.subplot(2, 1, 1)
x1, y1 = create_lists(s1)
plt.plot(x1, y1, label="15 Epoker")
x2, y2 = create_lists(s2)
plt.plot(x2, y2, label="10 Epoekr")
x3, y3 = create_lists(s3)
plt.plot(x3, y3, label="5 Epoker")

y = find_list(y1, y2, y3, max([max(i) for i in [y1, y2, y3]]))
plt.text(10, 0.25, f'Högst träffsäkerhet då: $n = {x1[y.index(max(y))]}, trfs.= {max(y)}$', fontsize = 10, 
         bbox = dict(facecolor="white", alpha = 1))

plt.title("Ett lager med 1-40 dolda neuroner, Sigmoid och CE")
plt.ylabel("Modellens träffsäkerhet")
plt.legend()
plt.yticks([(i+2)/10 for i in range(9)])

plt.subplot(2, 1, 2)
x4, y4 = create_lists(s4)
plt.plot(x4, y4, label="15 Epoker")
x5, y5 = create_lists(s5)
plt.plot(x5, y5, label="10 Epoker")
x6, y6 = create_lists(s6)
plt.plot(x6, y6, label="5 Epoker")

y = find_list(y4, y5, y6, max([max(i) for i in [y4, y5, y6]]))
plt.text(10, 0.25, f'Högst träffsäkerhet då: $n = {x1[y.index(max(y))]}, trfs.= {max(y)}$', fontsize = 10, 
         bbox = dict(facecolor="white", alpha = 1))

plt.title("Ett lager med 1-40 dolda neruoner. ReLU och CE")
plt.xlabel("Antal dolda neuroner")
plt.ylabel("Modellens träffsäkerhet")
plt.legend()
plt.yticks([(i+2)/10 for i in range(9)])
plt.show()"""


l = s.split("\n")

for i, e in enumerate(l):
    print(f"{i+1}           {e}")