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
from math import gamma, exp, factorial

import matplotlib.pyplot as plt
import json
import time

#
# spread methods
#

# If the contagion has an initially probability of spreading p, we will assume that
# subsequent "generations" of the contagion have a pth-power chance of spreading.


def fixed_spread(p,m):
    return p

def cos_spread(p,m):
    return p ** (m + cos(m*pi/2))

def exp_spread(p, m):
    return p ** m


#
# simulation methods
#

# Given an initial probability of spreading p and a function prob_spread
# that specifies the probability of spreading in each generation,
# sim_contagion(...) simulates one contagion.  We use numpy.random for 
# random number generation.
def sim_contagion(p, prob_spread=exp_spread):
    marks = collections.Counter({1: 1})
    n = 1
    while marks[n] > 0:  # while there are enemies still alive
        for _ in range(marks[n]):
            roll = rng.random()
            if roll < prob_spread(p, n):
                marks[n + 1] += 3
        n += 1
    return marks


# batch_simulate calls sim_contagion multiple times and tabulates the 
# results in a dictionary data.  If a previously tabulated dictionary
# is passed, the method will simply add to the previous simulations.
def batch_simulate(p, data=None, prob_spread=exp_spread, n=1000):
    if data is None:
        data = {"Cont": [], "Stop": [], "p": float(p), "spread": str(prob_spread), "n": int(n)}
    elif p != data["p"] or str(prob_spread) != data["spread"]:
        print "Error:  Mismatched parameter values"
        return data
    else:
        data["n"] += int(n)
    for _ in range(1, n + 1):
        marks = sim_contagion(p, prob_spread)
        roll_stop = np.max(marks.keys())
        roll_cont = (sum(marks.values()) - 1) / 3
        data["Stop"].append(roll_stop)
        data["Cont"].append(roll_cont)
    return data


# 
# Data methods
# 


# Uses JSON to save the dictionary returned by batch_simulate
def save_dict(data):
    now = str(int(time.time() * 100))
    f = open('/Users/thomtyrrell/Documents/Code/GitHub/Diablo-3-Contagion/cont_dict' + now + '.txt', 'a')
    json.dump(data, f)
    f.close()


# Saves just the contamination and depth data in a CSV format.
def export_csv(data):
    now = str(int(time.time() * 100))
    f = open('/Users/thomtyrrell/Documents/Code/GitHub/Diablo-3-Contagion/cont_data' + now + '.csv', 'a')
    f.write("Contaminations,Depth" + "\n")
    for i in range(data["N"]):
        f.write(str(data["Cont"][i]) + ',' + str(data["Hist"][i]) + "\n")
    f.close()


# Saves a TXT file that can be immediately imported into R as a data frame
# with the command read.table("cont_data.txt",header=TRUE,sep=";")
def R_export(n,p,prob_spread=exp_spread):
    now = str(int(time.time() * 100))
    f = open('/Users/thomtyrrell/Documents/Code/GitHub/Diablo-3-Contagion/cont_data' + now + '.txt', 'a')
    f.write("Levels;Contaminations;Depth" + "\n")
    for i in range(n):
        data = sim_contagion(p,prob_spread).values()
        f.write('c(' + str(data)[1:-1] + ');' + str(sum(data)/3) + ';' + str(len(data)) + "\n")
    f.close()


#
# visualization methods
#


def visual_scatter(data):
    n = data["n"]
    plt.close()
    plt.scatter(range(1, n + 1), data["Cont"], s=1, color="red", label="Contaminations during trial")
    plt.plot(range(1, n + 1), [np.mean(data["Cont"][:i]) for i in range(1,n + 1)], '-', color="purple", label="Mean")
    plt.xlabel("Trial")
    plt.ylabel("Number of contaminations during trial")
    plt.xlim(1, n)
    plt.ylim(1, np.max(data["Cont"]))
    plt.legend(shadow=True)
    plt.savefig(str(n) + ' trial graph.png')
    print "Mean number of contaminations over " + str(n) + " trials is " + str(np.mean(data["Cont"]))
    print "Standard Deviation of number of contaminations over " + str(n) + " trials is " + str(np.std(data["Cont"]))


# Currently useless
def visual_stop(data):
    n = data["n"]
    plt.close()
    stop_data = data["Stop"]
    plt.plot(range(1, n + 1), [np.mean(stop_data[:i]) for i in range(1,n + 1)], '-', color="purple", label="Mean")
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
    cmax = np.max(cont)
    plt.close()
    plt.hist(cont, bins=[s - .5 for s in range(0, cmax + 1)], normed=1, width=1, facecolor='red')
    plt.xlim(-1, cmax+1)
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.savefig('hist.png')


def visual_stop_hist(data):
    stop = np.array(data["Stop"])
    max_stop = np.max(stop)
    x = range(1, max_stop + 1)
    plt.close()

    plt.hist(stop, bins=[s - .5 for s in range(0,max_stop+1)], normed=1, width=1, facecolor='red')
    plt.xlim(0, max_stop + 1)
    plt.xlabel("Stopping time during trial")
    plt.ylabel("Portion of Trials")
    plt.savefig('hist.png')


#
# Approximation methods
#


def gamma_density_approx(data):
    data_mean = np.mean(data["Cont"])
    data_std = np.std(data["Cont"])
    theta = data_std ** 2 / data_mean
    k = data_mean / theta
    gamma_density = lambda x: 1 / (gamma(k) * theta ** k) * x ** (k - 1) * exp(-x / theta)
    return gamma_density


def visual_gamma_approx(data):
    cont = np.array(data["Cont"])
    max_cont = np.max(cont)
    x = np.linspace(0, max_cont + 1)
    gam = gamma_density_approx(data)
    y = np.array([gam(c) for c in x])
    plt.close()

    plt.hist(cont, bins=[s - .5 for s in range(0,max_cont+1)], normed=1, width=1, facecolor='red', label="Data")
    plt.plot(x, y, '-', color="purple", label="Gamma Distribution")

    plt.xlim(-1, max_cont + 1)
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.legend(shadow=True, loc=1)

    plt.savefig('hist.png')


def visual_poisson_approx(data):
    cont = np.array(data["Cont"])
    max_cont = np.max(cont)
    x = np.linspace(1, max_cont + 1)
    lamb = expect(data["p"])
    dens = lambda k: lamb ** k * exp(-lamb) / gamma(k+1)
    y = np.array([dens(c) for c in x])
    plt.close()

    plt.hist(cont, bins=[s - .5 for s in range(0,max_cont+1)], normed=1, width=1, facecolor='red', label="Data")
    plt.plot(x, y, '-', color="purple", label="Poisson Distribution")

    plt.xlim(0, max_cont + 1)
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.legend(shadow=True, loc=1)

    plt.savefig('hist.png')