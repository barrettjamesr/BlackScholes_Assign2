import numpy as np

#(a)
def stdNorm(size):
    mu, sigma = 0, 1 # mean and standard deviation
    return np.random.normal(mu, sigma, size)

#(c)
def generateZed(ex, wai, rho):
    return rho * ex + (1-rho **2)**0.5 * wai

def main():
    #(b)
    X = stdNorm(200)
    Y = stdNorm(200)
    
    Z = generateZed(X, Y, 0.5)
    
    new_path = "C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\Q2-2_Results.txt"
    data = open(new_path, "w")
    print(X, file = data)
    print(Y, file = data)
    print(Z, file = data)
    print(np.corrcoef(X,Z))
    print(np.corrcoef(X,Z), file= data)
    
    data.close()
    
    
main()
