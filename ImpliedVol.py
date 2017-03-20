import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm

def FullBlackScholesCall(S, X, r, t, vol, q):
    d1 = (np.log(float(S)/float(X))+(float(r)-float(q)+float(vol)**2 / 2)*float(t))/(float(vol)*float(t)**0.5)
    d2 = d1 - float(vol) * float(t) ** 0.5
    
    return float(S) / np.exp(float(q) * float(t)) * norm.cdf(d1) - norm.cdf(d2) * float(X) / np.exp(float(r) * float(t))

def FullBlackScholesPut(S, X, r, t, vol, q):
    d1 = (np.log(float(S)/float(X))+(float(r)-float(q)+float(vol)**2 / 2)*float(t))/(float(vol)*float(t)**0.5)
    d2 = d1 - float(vol) * float(t) ** 0.5
    
    return norm.cdf(0-d2) * float(X) / np.exp(float(r) * float(t)) -float(S) / np.exp(float(q) * float(t)) * norm.cdf(0-d1)

def ImpliedVolatility(S, X, r, t, vol, q):
    
    
    return 0


def main():
    #read instrument data
    
    #f = open("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\instruments.csv", "r")
    #reader = csv.reader(f)
    #next(reader, None)
    #securities = list(reader)
    #f.close()
    securities = pd.read_csv("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\instruments.csv")
    
    #read price data
    #f = open("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\marketdata.csv", "r")
    #reader = csv.reader(f)
    #next(reader, None)
    #market_data = list(reader)
    #f.close()
    market_data = pd.read_csv("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\marketdata.csv")
    market_data.setIndex('LocalTime')
    
    #print(securities)
    #print(market_data)
    # process data into 31, 32 & 33


    # get most recent price for the minute

    # join with instrument details

    
    #calculate implied vol


    #save results to \31.csv", \32.csv", and \33.csv"


    #create graphs

    

main()





