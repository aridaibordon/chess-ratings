# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as sc

mpl.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['font.size']       = 12
plt.rcParams['axes.linewidth']  = 1

class read_fide:
    
    def __init__(self):

        with open('standard_rating_list.txt', 'r') as f:
    
            data = np.empty((4, 362189), dtype=object)
    
            k   = 0
    
            for line in f:
                
                if k == 0:
                    k += 1
                    continue
        
                data[0][k-1] = int(line[113:117]) # Rating
                data[1][k-1] = int(line[126:130]) # Birthday
                data[2][k-1] = line[76:79]        # Country
                data[3][k-1] = line[80:81]        # Sex
        
                k += 1
        
        self.data       = data
        self.rat        = data[0]
        
        self.lab_sex = self.data[3]=='F'
        
        self.list_cou   = np.sort(list(set(data[2])))
        
    def return_data(self):
        return self.data
    
    def global_data(self):
        
        # Get data from top 20 countries with more players
        
        counter = 0
        
        data = np.empty((4, 20), dtype=object)
                   
        for cou in self.list_cou:
            
            lab_cou = self.data[2]==cou       
            
            pob     = len(self.rat[lab_cou])
                               
            if pob >= 4050:            
                
                per = (len(self.rat[lab_cou*self.lab_sex]))
                
                data[0][counter] = cou
                data[1][counter] = pob
                data[2][counter] = round(per/pob, 2)
                
                print(r'Country:                 %s'   % cou)
                print(r'Total number of players: %i'   % pob)
                print('Female players (%%):      %.2f' % round(100*per/pob, 1))
                print('\n')
                print('Differences:')
                
                diff = np.empty(3)
                
                for k in range(3):
                    
                    top = [1, 20, 100]
                    
                    diff[k] = self.compare_differences(cou=cou, top=top[k])
                    print('%i (Top %i)' % (int(diff[k]), int(top[k])))
                
                data[3][counter] = diff
                
                print('\n')
                counter += 1
                        
        return data
    
    def bar_players(self):
        
        # Bar graph of top 20 countries
        
        data = self.global_data()
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.bar(range(20), data[1], width=.7, alpha=.6,
               label='Total rated players')
        ax.bar(range(20), data[2]*data[1], width=.7, alpha=.7,
               color='green', label='Female rated players')
        
        plt.ylabel('Number of players')
        
        ax.set_xticks(range(20))
        ax.set_xticklabels(data[0], fontsize=10)
        ax.legend(loc='upper left', fontsize=12)
    
    def fidedata_cou(self, cou):
        
        # Get global information an gaussian fit of a given country
        
        lab_bir = self.data[1]<=2000
        lab_cou = self.data[2]==cou
        lab_sex = self.data[3]=='F'
                      
        label   = lab_bir*lab_cou
        
        (mu, sigma)     = sc.norm.fit(self.rat[label])
        
        pop             = len(self.rat[label])
        ratio           = len(self.rat[label*lab_sex]) / len(self.rat[label])
        
        print(r'Normal fit for %s: %i, %i' % (cou, mu, sigma))
        
        return int(mu), int(sigma), pop, round(ratio, 3)
        
    def hist_country(self, cou):
        
        # Plot ELO histogram and gaussian fit for a given country
        
        lab_bir = self.data[1]<=2000
        lab_cou = self.data[2]==cou
                      
        label   = lab_bir*lab_cou
        
        fig, ax = plt.subplots(figsize=[6, 6])
        
        n, bins, patches = ax.hist(self.rat[label], histtype='step', bins=80,
                                   label='FIDE data')
        
        #plt.hist(self.rat[label*self.lab_sex], histtype='step', bins=80,
        #         normed=1)
        
        (mu, sigma)     = sc.norm.fit(self.rat[label])
        y               = mpl.mlab.normpdf(bins.astype(int), mu, sigma)
    
        ax.plot(bins, 1.2*sum(bins)*y, 'r--', linewidth=3,
                 label = r'Normal fit: $\mu = %i,\ \sigma = %i$' % (int(mu), int(sigma)))
        
        plt.xlim(mu-2.5*sigma, mu+2.5*sigma)
        plt.ylim(0, 320)
        
        plt.xlabel('ELO rating')
        plt.ylabel('Number of players')
        
        handles, labels = ax.get_legend_handles_labels()
        
        plt.legend(handles[::-1], labels[::-1], fontsize=10, loc='upper left')
        
        ax.xaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', top='off')
        ax.yaxis.set_tick_params(which='major', size=7, width=1.5,
                                 direction='inout', right='off')
        
        plt.grid()
        
    def compare_differences(self, cou, top):
        
        lab_bir     = self.data[1] <= 2000
        lab_cou     = self.data[2] == cou
        
        lab_sexf    = self.data[3] == 'F'
        lab_sexm    = self.data[3] == 'M'
                           
        rat_men     = np.sort(self.rat[lab_bir*lab_cou*lab_sexm])[::-1]
        rat_wom     = np.sort(self.rat[lab_bir*lab_cou*lab_sexf])[::-1]
        
        diff        = np.mean(rat_men[:top]) - np.mean(rat_wom[:top])
        
        print('Difference (Top %i in %s): %i' % (top, cou, int(diff)))
        
        return diff

if __name__ == '__main__':
    
    # Example
    
    # read_fide().hist_country(cou='ITA')