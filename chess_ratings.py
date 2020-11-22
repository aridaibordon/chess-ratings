# -*- coding: utf-8 -*-

# Top male player:      Carlsen, Magnus (2862)
# Top female player:    Hou, Yifan (2658)

import numpy as np
import numpy.random as rd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as sc

from read_FIDE import read_fide

mpl.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['font.size']       = 12
plt.rcParams['axes.linewidth']  = 1

class chess_ratings:
    
    def __init__(self, cou='RUS', top=1):
        
        fide_data   = read_fide().fidedata_cou(cou=cou)
        
        self.mean   = fide_data[0]
        self.var    = fide_data[1]
        self.pop    = fide_data[2]
        self.rate   = fide_data[3]
        
        self.cou    = cou
        self.top    = top
                                
    def generate_population(self):
        
        # Generate population
        
        ratA = rd.normal(loc=self.mean, scale=self.var,
                         size=int(self.pop*(1-self.rate)))
        ratB = rd.normal(loc=self.mean, scale=self.var,
                         size=int(self.pop*self.rate))
        
        return np.asarray(ratA), np.asarray(ratB)

    def compare_differences_cou(self, rep=10000):
        
        # Compare expected and observed ELO gap for a country
        
        diff = np.empty(rep, dtype=object)
        
        fig, ax = plt.subplots(figsize=[6, 6])
        
        for i in range(rep):
            
            ratA, ratB = self.generate_population()
            
            ratA, ratB = np.sort(ratA)[::-1], np.sort(ratB)[::-1]
                        
            diff[i] = int(np.mean(ratA[:self.top])-np.mean(ratB[:self.top]))
        
        (mean, sigma) = sc.norm.fit(diff)
        
        ax.hist(diff, bins=120, alpha=.75, label='Simulated data',
                histtype='stepfilled')
        plt.xlim(mean-3.5*sigma, mean+3.5*sigma)
        plt.ylim(0, 400)
        
        diff_r      = read_fide().compare_differences(self.cou, self.top)
        
        plt.xlabel('ELO difference')
        plt.ylabel('Number of repetitions')
        
        ax.axvline(x=diff_r, color='k', linestyle='--', linewidth=2,
                   label='Observed difference')
        
        ax.xaxis.set_tick_params(which='major', size=7, width=1,
                                 direction='inout', top='off')
        ax.yaxis.set_tick_params(which='major', size=7, width=1,
                                 direction='inout', right='off')
        
        plt.legend(loc='upper left', fontsize=10)
        
    def comp_ratio(self, rep=1000):
        
        # ELO gap as a function of the ratio of women players
        # for a standard population
        
        mean    = 1500
        var     = 200
        size    = 5000
        
        data    = np.empty(25, dtype=object)
        
        for i in range(1, 26):
            
            print(i)
            
            data_rep = np.empty(rep)
            
            for rep in range(rep):
            
                ratA = rd.normal(loc=mean, scale=var,
                                 size= int(size*(1 - 0.015*i)))
                ratB = rd.normal(loc=mean, scale=var,
                                 size= int(size*0.015*i))
                
                ratA, ratB          = np.sort(ratA)[::-1], np.sort(ratB)[::-1]
                diff                = int(np.mean(ratA[:self.top])-np.mean(ratB[:self.top]))
                
                data_rep[rep]    = diff
            
            data[i-1] = data_rep
        
        mean    = np.asarray([np.mean(data[i]) for i in range(len(data))])
        std     = np.asarray([np.std(data[i])  for i in range(len(data))])/np.sqrt(rep)
        
        x       = np.arange(1.5, 38.51, 1.5)/100
                
        return x, mean, std
    
    def comp_popul(self, rep=1000):
        
        # ELO gap as a function of the total population
        # for a standard population
        
        mea    = 1500
        var    = 200
        rat    = .1
        
        mean    = np.empty(25)
        std     = np.empty(25)
        
        for i in range(1, 26):
            
            print(i)
            
            data_rep = np.empty(rep)
            
            for rep in range(rep):
                
                ratA = rd.normal(loc=mea, scale=var, size= int(2000*i*(1 - rat)))
                ratB = rd.normal(loc=mea, scale=var, size= int(2000*i*rat))
                
                ratA, ratB          = np.sort(ratA)[::-1], np.sort(ratB)[::-1]
                diff                = int(np.mean(ratA[:self.top])-np.mean(ratB[:self.top]))
                
                data_rep[rep]    = diff
            
            mean[i-1]   = np.mean(data_rep)
            std[i-1]    = np.std(data_rep) / np.sqrt(rep)
        
        x       = np.arange(2000, 50001, 2000)
                
        # Linear fit
        
        a, b    = np.polyfit(np.log(x), np.log(mean), 1)
        
        return x, mean, std, a, b
        
    def comp_varia(self, rep=1000):
        
        # ELO gap as a function of the variance
        # for a standard population
        
        mea    = 1500
        
        mean    = np.empty(26)
        std     = np.empty(26)
        
        for i in range(26):
            
            print(i+1)
            
            data_rep = np.empty(rep)
            
            for rep in range(rep):
            
                ratA = rd.normal(loc=mea, scale=16*i+200, size=4500)
                ratB = rd.normal(loc=mea, scale=16*i+200, size=500)
                
                ratA, ratB          = np.sort(ratA)[::-1], np.sort(ratB)[::-1]
                diff                = int(np.mean(ratA[:self.top])-np.mean(ratB[:self.top]))
                
                data_rep[rep]    = diff
            
            mean[i] = np.mean(data_rep)
            std[i]  = np.std(data_rep) / np.sqrt(rep)
        
        x = 24*np.arange(26) + 200
                
        # Linear fit
        
        a, b    = np.polyfit(x[:], mean[:], 1)
        
        return x, mean, std, a, b
    
    def global_comp(self, rep=1000):
        
        # Comparation of the normalized expected and observed difference
        # for the top 20 countries by number of players
        
        data        = read_fide().global_data()
        
        cou         = data[0]
        diff_fide   = np.asarray([data[3][i][0] for i in range(20)]) 
        data        = np.empty(20)
        
        for i in range(20):
            
            print(cou[i])
            
            fide_data   = read_fide().fidedata_cou(cou=cou[i])
        
            mean        = fide_data[0]
            var         = fide_data[1]
            pop         = fide_data[2]
            rate        = fide_data[3]
            
            diff = np.empty(rep, dtype=object)
    
            for j in range(rep):
            
                ratA = rd.normal(loc=mean, scale=var, size=int(pop*(1-rate)))
                ratB = rd.normal(loc=mean, scale=var, size=int(pop*rate))
                
                ratA, ratB = np.sort(ratA)[::-1], np.sort(ratB)[::-1]
                        
                diff[j] = int(np.mean(ratA[:self.top])-np.mean(ratB[:self.top]))
        
            data_fit    = sc.norm.fit(diff)
            
            data[i] = (diff_fide[i] - data_fit[0]) / data_fit[1]
            print(data[i])
        
        return data
    
    def plot_global_comp(self):
        
        data = self.global_comp()
    
        x           = np.linspace(-3, 3, 1000)
    
        fig, ax = plt.subplots(figsize=[6, 6])
    
        for i in data:
        
            ax.axvline(x=i, ymin=0, ymax=2*sc.norm.pdf(i, 0, 1), linestyle='-',
                       color=[.5, .5, .5], linewidth=.4)
    
        mean = np.mean(data)
    
        plt.axvline(x=mean, ymin=0, ymax=2*sc.norm.pdf(mean, 0, 1)-.005,
                    linewidth=2, color=[0,0,0], label=r'Normalized mean difference')
        plt.plot(x, sc.norm.pdf(x, 0, 1), linewidth=2.3, color='r',
                 label=r'Normal distribution with $\mu = 0$ and $\sigma = 1$')
        plt.xlim(min(x), max(x))
        plt.ylim(0, .5)
    
        handles, labels = ax.get_legend_handles_labels()
    
        plt.legend(handles[::-1], labels[::-1], fontsize=10, loc='upper left')
        
        ax.xaxis.set_tick_params(which='major', size=7, width=1,
                                 direction='inout', top='off')
        ax.yaxis.set_tick_params(which='major', size=7, width=1,
                                 direction='inout', right='off')

    def plot_var(self):
        
        x, mean, std, a, b = chess_ratings().comp_varia()
        
        y = a*x+b
    
        fig, ax = plt.subplots(figsize=[6, 6])
    
        label = r'Linear fit: $y = {0} x + {1}$'.format(round(a,1), round(b,2))
    
        ax.errorbar(x=x, y=mean, yerr=std, fmt='.', color=[.05, .48, .71],
                    label='Numerical results for a\nconstant population and ratio')
        ax.plot(x, y, label=label, color=[.94, .15, .15])
        
        plt.xlabel('Variance')
        plt.ylabel('ELO difference')
        plt.xlim(180, 820)
        plt.ylim(100, 400)
    
        ax.xaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', top='off')
        ax.yaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', right='off')
    
        handles, labels = ax.get_legend_handles_labels()
    
        plt.legend(handles[::-1], labels[::-1], fontsize=10, loc='upper left')
        
        plt.grid()

    def plot_pop(self):
        
        x, mean, std, a, b = self.comp_popul()
        
        x_fit   = np.linspace(0, max(x), 100)
    
        y       = np.exp(a*np.log(x_fit)+b)
        
        
        fig, ax = plt.subplots(figsize=[6, 6])
    
        label = 'Fitted curve'
        
        ax.errorbar(x=x, y=mean, yerr=std, fmt='.', color=[.05, .48, .71],
                    label='Numerical results for a\nconstant ratio and variance')
        ax.plot(x_fit, y, label=label, color=[.94, .15, .15])
    
        plt.xlabel('Total population')
        plt.ylabel('ELO difference')
        plt.xlim(0, 50000)
        plt.ylim(100, 140)
    
        ax.xaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', top='off')
        ax.yaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', right='off')
    
        handles, labels = ax.get_legend_handles_labels()
    
        plt.legend(handles[::-1], labels[::-1], fontsize=10, loc='upper right')
        
        plt.grid()

    def plot_rat(self):
        
        x, mean, std = self.comp_ratio()
        
        fig, ax = plt.subplots(figsize=[6, 6])
            
        ax.errorbar(x=x, y=mean, yerr=std, fmt='-', color=[.05, .48, .71],
                    label='Numerical results for a constant\npopulation and variance')
    
        plt.xlabel('Ratio (%)')
        plt.ylabel('ELO difference')
        plt.xlim(0, 0.4)
        plt.ylim(0, 300)
    
        ax.xaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', top='off')
        ax.yaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', right='off')
        
        plt.legend(fontsize=10, loc='upper right', numpoints=1)
        
        plt.grid()    

if __name__ == '__main__':
    
    chess_ratings().global_comp()
