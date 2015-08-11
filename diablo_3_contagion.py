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

def exp_spread(p,m): return p ** m

# sim_contagion(...) will simulate an outbreak and count the number of spreads.
def sim_contagion(p, prob_spread=exp_spread):
    marks = collections.Counter({1: 3})
    n = 1
    while marks[n] > 0:  # while there are enemies still alive
        for _ in range(marks[n]):
            roll = rng.random()
            if roll < prob_spread(p,n):
                marks[n + 1] += 3
        n += 1
    return sum(np.array(marks.values()) / 3)


def batch_simulate(p,prob_spread=exp_spread,N=100):
    data = {"Cont":[],"Mean":[],"Hist":collections.Counter(),"p":p,"spread":prob_spread,"N":N}
    for _ in range(1, N):
        roll_cont = sim_contagion(p,prob_spread)
        data["Cont"].append(roll_cont)
        data["Mean"].append(np.mean(data["Cont"]))
        data["Hist"][roll_cont]+=1
    return data


def visual_scatter(data):
    N = data["N"]
    plt.close()
    plt.scatter(range(1, N), data["Cont"], s=1, color="red", label="Contaminations during trial")
    plt.plot(range(1, N), data["Mean"], '-', color="purple", label="Mean")
    plt.xlabel("Trial")
    plt.ylabel("Number of contaminations during trial")
    plt.xlim(1,N)
    plt.ylim(1,np.max(sorted(data["Hist"].keys())))
    plt.legend(shadow=True)
    plt.savefig(str(N) + ' trial graph.png')
    print "Mean number of contaminations over " + str(N) + " trials is " + str(data["Mean"][-1]*1.)
    print "Standard Deviation of number of contaminations over " + str(N) + " trials is " + str(np.std(data["Cont"]))


def visual_hist(data):
    N = data["N"]
    cont = np.array(data["Cont"])
    keys = sorted(data["Hist"].keys())
    plt.close()
    plt.hist(cont, bins=[s-.5 for s in keys+[keys[-1]+3]], normed=1, width=1, facecolor='red')
    plt.xlim(0,np.max(keys))
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.savefig('hist.png')


def gamma_density_approx(data):
    data_mean = data["Mean"][-1]*1.
    data_std = np.std(data["Cont"])
    theta = (data_std) ** 2 / data_mean
    k = data_mean/theta
    gamma_density = lambda x: 1 / ( gamma(k) * theta ** k ) * x ** ( k - 1 ) * exp( -x / theta )
    return gamma_density
    
    
def visual_gamma_approx(data):
    cont = np.array(data["Cont"])
    max_cont = np.max(cont)
    keys = sorted(data["Hist"].keys())
    x = np.linspace(0,max_cont)
    gamma = gamma_density_approx(data)
    y = gamma(x)
    plt.close()

    plt.hist(cont, bins=[s-.5 for s in keys+[keys[-1]+3]], normed=1, width=1, facecolor='red', label="Data")
    plt.plot(x,y,'-',color="purple",label="Gamma Distribution")

    plt.xlim(0,max_cont)
    plt.xlabel("Number of contaminations during trial")
    plt.ylabel("Portion of Trials")
    plt.legend(shadow=True,loc=1)

    plt.savefig('hist.png')
    
    
def visual_expect(prob_spread=exp_spread,P=.75,N=100):
    x = np.linspace(0, P)
    y = np.array([batch_simulate(p,prob_spread,N)["Mean"][-1] for p in x])
    y_exp = expect(x)
    plt.close()
    plt.plot(x, y, 'x', color="red", label="Mean number of contaminations after " + str(N) + " trials")
    plt.plot(x, y_exp, '-', color="purple", label="Expected number of contaminations")
    plt.legend(shadow=True,loc=2)
    plt.xlabel("Probability of Spreading")
    plt.ylabel("Number of contaminations")
    plt.savefig(str(N) + ' trial graph.png')
    
    
def prob_x(n,s,p,prob_spread=exp_spread):
    if n == 1:
        return binomial(3,s) * p ** s * ( 1 - p ) ** ( 3 - s )
    else:
        return sum([binomial(3 * r, s) * prob_spread(p,n) ** s * ( 1 - prob_spread(p,n) ) ** ( 3 * r - s ) * prob_x(n-1,r,p,prob_spread) for r in range( int(np.ceil( s / 3 )) , 3 ** ( n - 1 ) + 1 ) ] )

def expect_x(n,p,prob_spread=exp_spread):
    return sum([s * prob_x(n,s,p,prob_spread) for s in range(3 ** n + 1)])

def expect(p,prob_spread=exp_spread,n_bar=150):
    return sum([ 3 ** n * p ** ((n ** 2 + n) / 2) for n in range(n_bar)])


