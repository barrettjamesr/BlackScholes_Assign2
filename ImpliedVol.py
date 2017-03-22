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
    securities = pd.read_csv("C:\\Users\\guest2\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\instruments.csv")
    securities.set_index('Symbol')
    
    #read price data
    #f = open("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\marketdata.csv", "r")
    #reader = csv.reader(f)
    #next(reader, None)
    #market_data = list(reader)
    #f.close()
    market_data = pd.read_csv("C:\\Users\\guest2\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\marketdata.csv")
    market_data.set_index('LocalTime')
    market_data['LocalTime'] = market_data['LocalTime'].apply(lambda x: 
                                    dt.datetime.strptime(x,'%Y-%b-%d %I:%M:%S.%f'))
    market_data = market_data.sort_values('LocalTime')

    data31 = market_data[(market_data['LocalTime'] < dt.datetime(2016, 2, 16, 9, 31, 0, 0))]
    data32 = market_data[(market_data['LocalTime'] < dt.datetime(2016, 2, 16, 9, 32, 0, 0))]
    data33 = market_data[(market_data['LocalTime'] < dt.datetime(2016, 2, 16, 9, 33, 0, 0))]

    #data31 = data31.groupby("Symbol").apply(lambda d:d.loc[d.LocalTime.idxmax()])

    data31 = data31[['LocalTime', 'Symbol', 'Last', 'Bid1', 'Ask1']].groupby('Symbol').last()

    combined31 = pd.concat([securities, data31.reset_index()], axis=1)
    

    #print(securities)
    #print(market_data)
    # process data into 31, 32 & 33

    #print(data31.reset_index())
    print(combined31)

    # get most recent price for the minute

    # join with instrument details

    
    #calculate implied vol


    #save results to \31.csv", \32.csv", and \33.csv"


    #create graphs

    

main()





