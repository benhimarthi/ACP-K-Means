import numpy as np
import matplotlib.pyplot as plt

class Standardisator:
    def __init__(self, variable):
        self.var = variable
        
    @property
    def centralization(self):
        uno = np.ones((1, len(self.var)))
        p1 = self.var - np.dot(np.dot(np.transpose(uno), uno), self.var)/len(self.var)
        return p1
    
    @property
    def covariance_matrix(self):
        mat = self.centralization
        return np.dot(np.transpose(mat), mat)/len(mat)
    
    @property
    def varianceCalculation(self):
        shape = np.shape(self.var)
        vart = []
        for n in range(shape[1]):
            vart.append(Standardisator.VARIANCE(Standardisator.EXTRACT_COLUMN(self.var, n)))
        return vart
    
    @property
    def standardisation(self):
        self.var = self.centralization
        variances = self.varianceCalculation
        for n in range(self.var.shape[0]):
            for m in range(self.var.shape[1]):
                if m==0:
                    self.var[n, m] = self.var[n, m]/np.sqrt(variances[0])
                else : self.var[n, m] = self.var[n, m]/np.sqrt(variances[1])
        return self.var

    @property
    def diagonalization(self):
        return np.linalg.eigvals(self.covariance_matrix)

    @property
    def proper_vector(self):
        return np.linalg.eig(self.covariance_matrix)

    @property
    def information_quantity(self):
        return self.diagonalization/self.covariance_matrix.trace()
    
    @property
    def rotatedMatrixe(self):
        pass
    
    @staticmethod
    def REMOVE_COLUMN_FROM_MATRIX(mat, *colum):
        res = []
        shape = mat.shape
        print(colum)
        for n in range(shape[0]):
            cl = []
            for m in range(shape[1]):
                if m not in colum : cl.append(mat[n, m])
            res.append(cl)
        return np.array(res)
                   
    @staticmethod
    def VARIANCE(lst):
        sm = 0
        m = Standardisator.AVARAGE(lst)
        for n in lst:
            sm = sm + (n - m)**2
        return sm/len(lst)
    
    @staticmethod
    def EXTRACT_COLUMN(mt, cl):
        shape = mt.shape
        t = []
        for n in range(shape[0]):
            t.append(mt[n, cl])
        return t
            
    @staticmethod
    def AVARAGE(lst):
        return sum(lst)/len(lst)
    
    

def main():
    matrix = np.array([[2, 2, 3], [3, 1, 2], [1, 0, 3], [2, 1, 4], [2, 1, 3]])
    
    matrix = np.sqrt(10) * matrix
    acp = Standardisator(matrix)
    matrix_standard_form = acp.standardisation
    correlation_matrix = acp.covariance_matrix
    variables_information_quantity = acp.information_quantity
    proper_vector = acp.proper_vector
    reducted_data = np.dot(matrix_standard_form, Standardisator.REMOVE_COLUMN_FROM_MATRIX(proper_vector[1], 0))
    plt.plot(reducted_data, 'b.')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()