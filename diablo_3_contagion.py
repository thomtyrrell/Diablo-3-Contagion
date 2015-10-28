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
from numpy import random as rng

import matplotlib.pyplot as plt
import json
import time

from scipy.misc import comb
from math import gamma, exp, factorial
from statistics import mean


# probability methods


def exp_spread(p, m):
    return p ** m


def prob_x(n, s, p, prob_spread=exp_spread):
    prob = prob_spread(p, (n ** 2 + n) / 2)
    return comb(3 ** n, s, exact=True) * prob ** s * (1 - prob) ** (3 ** n - s) * 1.0


# exponential spread methods


def expect_x(n, p):
    return 3 ** n * p ** ((n ** 2 + n) / 2)


def prob_s(n, p):
    return (prob_x(n, 0, p) - prob_x(n - 1, 0, p)) / (1 - prob_x(n - 1, 0, p))


def expect(p, n_bar=150):
    return sum([3 ** n * p ** ((n ** 2 + n) / 2) for n in range(n_bar)])


def var(p, n_bar=150):
    return sum([3 ** n * p ** ((n ** 2 + n) / 2) * (1 - p ** ((n ** 2 + n) / 2)) for n in range(n_bar)])


#
# simulation methods
#

# Given such a counter marks and initial probability of spreading p,
# kill_enemy(...) randomly chooses an enemy to kill, and spreads the mark
# probabilistically.  We use numpy.random for random number generation.
# Currently randomly chooses an enemy to kill

# sim_contagion(...) will simulate an outbreak and count the number of spreads.
def sim_contagion(p, prob_spread=exp_spread):
    marks = collections.Counter({1: 3})
    n = 1
    while marks[n] > 0:  # while there are enemies still alive
        for _ in range(marks[n]):
            roll = rng.random()
            if roll < prob_spread(p, n):
                marks[n + 1] += 3
        n += 1
    return marks


def batch_simulate(p, data=None, prob_spread=exp_spread, n=100):
    if data is None:
        data = {"Cont": [], "Stop": [], "Hist": collections.Counter(), "StopHist": collections.Counter(), "p": float(p),
                "spread": str(prob_spread), "n": int(n)}
    elif p != data["p"] or str(prob_spread) != data["spread"]:
        print "Error:  Mismatched parameter values"
        return data
    else:
        data["n"] += int(n)
    for _ in range(1, n + 1):
        marks = sim_contagion(p, prob_spread)
        roll_stop = np.max(marks.keys())
        data["Stop"].append(roll_stop)
        data["StopHist"][roll_stop] += 1
        roll_cont = sum(np.array(marks.values()) / 3)
        data["Cont"].append(roll_cont)
        data["Hist"][roll_cont] += 1
    return data


def save_dict(data):
    now = str(int(time.time() * 100))
    f = open('/Users/thomtyrrell/Documents/Code/GitHub/Diablo-3-Contagion/cont_dict' + now + '.txt', 'a')
    json.dump(data, f)
    f.close()


def export_csv(data):
    now = str(int(time.time() * 100))
    f = open('/Users/thomtyrrell/Documents/Code/GitHub/Diablo-3-Contagion/cont_data' + now + '.csv', 'a')
    f.write("Contaminations,Depth" + "\n")
    for i in range(data["N"]):
        f.write(str(data["Cont"][i]) + ',' + str(data["Stop"][i]) + "\n")
    f.close()


#
# visualization methods
#

def visual_scatter(data):
    n = data["n"]
    plt.close()
    plt.scatter(range(1, n + 1), data["Cont"], s=1, color="red", label="Contaminations during trial")
    plt.plot(range(1, n + 1), [np.mean(data["Cont"])] * n, '-', color="purple", label="Mean")
    plt.xlabel("Trial")
    plt.ylabel("Number of contaminations during trial")
    plt.xlim(1, n)
    plt.ylim(1, np.max(sorted(data["Hist"].keys())))
    plt.legend(shadow=True)
    plt.savefig(str(n) + ' trial graph.png')
    print "Mean number of contaminations over " + str(n) + " trials is " + str(np.mean(data["Cont"]))
    print "Standard Deviation of number of contaminations over " + str(n) + " trials is " + str(np.std(data["Cont"]))


def visual_stop(data):
    n = data["n"]
    stop_data = data["Stop"]
    plt.close()
    plt.plot(range(1, n + 1), [np.mean(stop_data)] * n, '-', color="purple", label="Mean")
    plt.scatter(range(1, n + 1), stop_data, s=1, color="red")
    plt.xlabel("Trial")
    plt.ylabel("Stopping Time")
    plt.xlim(1, n)
    plt.ylim(1, np.max(stop_data))
    plt.legend(shadow=True)
    plt.savefig(str(n) + ' trial graph.png')
    print "Mean stopping level over " + str(n) + " trials is " + str(np.mean(stop_data))


def visual_hist(data):
    cont = np.array(data["Cont"])
    keys = sorted(data["Hist"].keys())
    plt.close()
    plt.hist(cont, bins=[s - .5 for s in keys + [keys[-1] + 3]], normed=1, width=1, facecolor='red')
    plt.xlim(0, np.max(keys))
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.savefig('hist.png')


def visual_stop_hist(data):
    stop = np.array(data["Stop"])
    keys = sorted(data["StopHist"].keys())
    max_stop = np.max(stop)
    x = range(1, max_stop + 1)
    fun = lambda t: prob_s(t, data["p"])
    y = [fun(n) for n in x]
    plt.close()

    plt.hist(stop, bins=[s - .5 for s in keys + [keys[-1] + 3]], normed=1, width=1, facecolor='red')
    plt.plot(x, y, '-', color="purple", label="Stop Distribution")

    plt.xlim(0, max_stop + 1)
    plt.xlabel("Stopping time during trial")
    plt.ylabel("Portion of Trials")
    plt.legend(shadow=True)
    plt.savefig('hist.png')


def gamma_density_approx(data):
    data_mean = np.mean(data["Cont"])
    data_std = np.std(data["Cont"])
    theta = data_std ** 2 / data_mean
    k = data_mean / theta
    gamma_density = lambda x: 1 / (gamma(k) * theta ** k) * x ** (k - 1) * exp(-x / theta)
    return gamma_density


def visual_gamma_approx(data):
    cont = np.array(data["Cont"])
    max_cont = int(np.max(cont))
    keys = sorted(data["Hist"].keys())
    x = np.linspace(0, max_cont + 1)
    gam = gamma_density_approx(data)
    y = gam(x)
    plt.close()

    plt.hist(cont, bins=[s - .5 for s in keys + [keys[-1] + 3]], normed=1, width=1, facecolor='red', label="Data")
    plt.plot(x, y, '-', color="purple", label="Gamma Distribution")

    plt.xlim(0, max_cont + 1)
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.legend(shadow=True, loc=1)

    plt.savefig('hist.png')


def visual_poisson_approx(data):
    cont = np.array(data["Cont"])
    max_cont = np.max(cont)
    keys = sorted(data["Hist"].keys())
    # noinspection PyTypeChecker
    x = np.arange(0, max_cont + 1)
    lamb = expect(data["p"], data["spread"])
    dens = lambda k: lamb ** k * exp(-lamb) / factorial(k)
    y = [dens(n) for n in x]
    plt.close()

    plt.hist(cont, bins=[s - .5 for s in keys + [keys[-1] + 3]], normed=1, width=1, facecolor='red', label="Data")
    plt.plot(x, y, '-', color="purple", label="Poisson Distribution")

    plt.xlim(0, max_cont + 1)
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.legend(shadow=True, loc=1)

    plt.savefig('hist.png')


def visual_expect(prob_spread=exp_spread, p_max=.75, n=100):
    x = np.linspace(0, p_max)
    # noinspection PyTypeChecker
    y = np.array([mean(batch_simulate(p, prob_spread, n)["Cont"]) for p in x])
    y_exp = np.array([expect(p) for p in x])
    plt.close()
    plt.plot(x, y, 'x', color="red", label="Mean number of contaminations after " + str(n) + " trials")
    plt.plot(x, y_exp, '-', color="purple", label="Expected number of contaminations")
    plt.legend(shadow=True, loc=2)
    plt.xlabel("Probability of Spreading")
    plt.ylabel("Number of contaminations")
    plt.savefig(str(n) + ' trial graph.png')
