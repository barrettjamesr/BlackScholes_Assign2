import numpy as np
from scipy.stats import norm

def BlackScholesCall(S, X, r, t, vol):
    d1 = (np.log(float(S)/float(X))+(float(r)+float(vol)**2 / 2)*float(t))/(float(vol)*float(t)**0.5)
    d2 = d1 - float(vol) * float(t) ** 0.5
    
    return float(S) * norm.cdf(d1) - norm.cdf(d2) * float(X) / np.exp(float(r) * float(t))

def BlackScholesPut(S, X, r, t, vol):
    d1 = (np.log(float(S)/float(X))+(float(r)+float(vol)**2 / 2)*float(t))/(float(vol)*float(t)**0.5)
    d2 = d1 - float(vol) * float(t) ** 0.5
    
    return norm.cdf(0-d2) * float(X) / np.exp(float(r) * float(t)) -float(S) * norm.cdf(0-d1)

def main():
    S = input("Stock Price:")
    X = input("Strike Price:")
    r = input("risk free rate:")
    t = input("time to maturity:")
    vol = input("volatility:")

    print ("Call price is " + str(BlackScholesCall(S, X, r, t, vol)))
    print ("Put price is " + str(BlackScholesPut(S, X, r, t, vol)))
           
    
main()

