import numpy as np
import matplotlib.pyplot as plt
import random as rd
from ACP import Standardisator as std

class K_means:
    def __init__(self, variable, center_nb, centers = None):
        self.variables = self.__standardisation(variable)
        self.__centers_nb = center_nb
        self.centers = self.__choose_random_center() if centers == None else list(map(lambda x : list(self.variables[x]), centers))
        self.group = {}
        
    def __standardisation(self, var):
        res = std(var)
        return res.standardisation()
    
    def __choose_random_center(self) :
        n = 0
        center = []
        while n < self.__centers_nb:
            index = rd.random() * len(self.variables)
            index = int(index)
            if list(self.variables[index]) not in center:
                center.append(list(self.variables[index]))
                n += 1
        return center
                
    def euc_distance(self, pt1, pt2):
        pt2 = np.array(pt2)
        dst = pt1 - pt2
        dst = dst**2
        return np.sqrt(sum(dst))

    def generate_group(self):
        group = {}
        #Genretae groups
        for n in range(self.__centers_nb):
            group[n] = []
        for m in self.variables:
            dst_set = []
            for i in self.centers :
                dst_set.append(self.euc_distance(m, i))
            t = min(dst_set)
            group[dst_set.index(t)].append(m)
        return group
    
    def calculate_new_centers(self):
        datas = self.generate_group()
        self.group = datas
        for n in enumerate(datas):
            self.centers[n[0]] = self.calculate_center_from_tab(datas[n[1]])
        
    def calculate_center_from_tab(self, dt):
        dt = np.array(dt)
        res = []
        for n in range(dt.shape[1]):
             currentCol = std.EXTRACT_COLUMN(dt, n)
             res.append(std.AVARAGE(currentCol))
        return res
    
             
def main():
    x = K_means(np.array([[1, 3], [3, 3], [4, 3], [5, 3], [1, 2], [4, 2], [1, 1], [2, 1]]),2)
    old_center = x.centers
    new_center = []
    while(new_center != old_center):
        old_center = new_center
        new_center = x.calculate_new_centers()
        
    print(x.centers)
    plt.plot(x.centers[0], 'g.')
    plt.plot(x.centers[0], 'y.')
    for n in enumerate(x.group):
        plt.plot(x.group[n[1]], "b." if n[0]==0 else "r.")
        plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
