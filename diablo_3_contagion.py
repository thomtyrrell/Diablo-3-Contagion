# Diablo 3 Analysis
# Marked for Death ability - Marked enemy takes 20% extra damage for 15 seconds
# Contagion rune - On death of first marked enemy, mark spreads to 3 nearest enemies within 30 yards.
# Further deaths have a chance to spread to another 3 enemies.
# We will assume there is no shortage of enemies for the mark to spread to.

# p(n) = probability mark will spread on nth opportunity
# should decay to 0 as n approaches infinity
# We will assume exponential decay by a constant rate
# So, p(n) = p^n, where p = p(1) the initial chance to spread

# We will model each outbreak using a counter.  The keys will count the iteration
# of a mark. 

import collections
import numpy as np
import scipy as sc
from numpy import random as rng
import matplotlib.pyplot as plt

# Given such a counter marks and initial probability of spreading p,
# kill_enemy(...) randomly chooses an enemy to kill, and spreads the mark
# probabilistically.  We use numpy.random for random number generation.
# Currently randomly chooses an enemy to kill

# sim_contagion(...) will simulate probabilistically the number of marks in one outbreak.
def sim_contagion(p):
    marks = collections.Counter({1: 3})
    cont = 1
    while len(marks.keys()) > 0:  # while there are enemies still alive
        for m in marks.keys():
            if marks[m] == 1:
                del marks[m]
            else:
                marks[m] -= 1
            roll = rng.random()
            if roll < p ** m:
                marks[m + 1] += 3
                cont += 1
    return cont


def batch_simulate(p,N=100):
    data = {"Cont":[],"Mean":[],"Hist":collections.Counter(),"p":p,"N":N}
    for _ in range(1, N):
        roll_cont = sim_contagion(p)
        data["Cont"].append(roll_cont)
        data["Mean"].append(np.mean(data["Cont"]))
        data["Hist"][roll_cont]+=1
    return data


def visualize_scatter(data):
    N = data["N"]
    plt.close()
    plt.scatter(range(1, N), data["Cont"], s=1, color="red", label="Contaminations by Trial")
    plt.plot(range(1, N), data["Mean"], '-', color="purple", label="Mean")
    plt.xlabel("Trial")
    plt.ylabel("Number of Contaminations")
    plt.xlim(1,N)
    plt.ylim(1,np.max(sorted(data["Hist"].keys())))
    plt.legend(shadow=True)
    plt.savefig(str(N) + ' trial graph.png')
    print "Mean number of contaminations over " + str(N) + " trials is " + str(data["Mean"][-1]*1.)


def visualize_hist(data):
    N = data["N"]
    cont = np.array(data["Cont"])
    keys = sorted(data["Hist"].keys())
    plt.close()
    plt.hist(cont, bins=[s-.5 for s in keys+[keys[-1]+3]], normed=1, width=1, facecolor='red')
    plt.xlim(0,np.max(keys))
    plt.xlabel("Number of contaminations over " + str(N) + " trials")
    plt.ylabel("Portion of Trials")
    plt.savefig('hist.png')


def visual_expect(P=.75,N=100):
    x = np.linspace(0, P)
    y = np.array([batch_simulate(p,N)["Mean"][-1] for p in x])
    y_exp = np.array([expect(p) for p in x])
    plt.close()
    plt.plot(x, y, 'x', color="red", label="Mean number of contaminations after " + str(N) + " trials")
    plt.plot(x, y_exp, '-', color="purple", label="Expected number of contaminations")
    plt.legend(shadow=True)
    plt.xlabel("Probability of Spreading")
    plt.ylabel("Number of contaminations")
    plt.savefig(str(N) + ' trial graph.png')

# find formula for expected number of spreads
# Theta functions?
expect = lambda p, n_bar=150: sum([p ** ((n ** 2 - n) / 2) * 3 ** (n-1) for n in range(1,n_bar)])
mass = lambda p, n: 3 ** (n - 1) * p ** ((n ** 2 - n)/2)
poisson = lambda lam, k: lam ** k / factorial(k) * np.exp(-lam)
