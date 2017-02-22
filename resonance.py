#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import itertools
import numpy as np
from operator import itemgetter

def Program():
    font = { 'family' : 'sans-serif', 'weight' : 'bold', 'size' : 12 }
    plt.rc('font', **font)
    fig, ax = plt.subplots(facecolor='white')

    # Some dummy data points to fit against
    x = np.array([ 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 
        5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9 ])
    y = np.array([ 1000.0, 700.0, 500.0, 300.0, 250.0, 170.0, 160.0, 140.0, 
        160.0, 500.0, 20.0, 120.0, 10.0, 5.0, 3.0, 1.8, 1.6 ])

    p = np.poly1d(np.polyfit(np.log10(x), y, len(x)))

    # Plot the resonance fit
    #plt.plot(np.log10(x/1000.0), np.log10(y), '.', np.log10(x/1000.0),np.log10(p(np.log10(x))), '-') 

    # Stretch out the curve we're interested in [10.0e-3, 10.0e6]
    energy = np.linspace(0, 1, num=30000)
    energy = 10.0e7*np.power(energy, 4)+10.0e-3

    cs = np.linspace(0, 1, num=30000)
    cs = 1.0e3*cs+1.0

    plt.plot(energy, cs)

    print energy, cs

    # Plot the energy
    #xtemp = np.linspace(1, 30000, num=30000)
    #plt.plot(xtemp, energy, '-')

    # Have to shift up the current x position to fit curve correctly so
    # multiplying by 1000
    with open('new_resonance.cs', 'w') as f:
        for rr in range(1, len(energy)):
            f.write("%.12e %.12e\n" % (energy[rr], cs[len(energy)-rr]))

    #ax.set_xlim([-2, 6])
    ax.grid(zorder=0)
    plt.title('Dummy resonance graph for lookup tables')
    plt.ylabel('Cross Section (barns)', fontsize=13)
    plt.xlabel('Particle Energy (eV)', fontsize=13)
    plt.show()

Program()
