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

def ImpliedVolatilityCall(S, X, r, t, vol, q):
    
    
    return 0

def ImpliedVolatilityPut(S, X, r, t, vol, q):
    
    
    return 0


def main():
    #constants
    r = 0.04
    q = 0.20

    #read instrument data
    securities = pd.read_csv("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\instruments.csv", dtype = str)
    securities = securities.set_index('Symbol')
    securities['Expiry'] = pd.to_datetime(securities['Expiry'], errors='coerce')

    #securities['Expiry'] = securities['Expiry'].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))
    #print(securities)

    #read price data
    market_data = pd.read_csv("C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\marketdata.csv", dtype = str)
    market_data['LocalTime'] = pd.to_datetime(market_data['LocalTime'], errors='coerce')
    #market_data['LocalTime'] = market_data['LocalTime'].apply(lambda x: dt.datetime.strptime(x,'%Y-%b-%d %I:%M:%S.%f'))
    market_data = market_data.sort_values('LocalTime')

    # process data into 31, 32 & 33
    data31 = market_data[(market_data['LocalTime'] < dt.datetime(2016, 2, 16, 9, 31, 0, 0))]
    data32 = market_data[(market_data['LocalTime'] < dt.datetime(2016, 2, 16, 9, 32, 0, 0))]
    data33 = market_data[(market_data['LocalTime'] < dt.datetime(2016, 2, 16, 9, 33, 0, 0))]

    # get most recent price for the minute
    data31 = data31.iloc[data31.groupby('Symbol')['LocalTime'].idxmax().values.ravel()]
    data32 = data32.iloc[data31.groupby('Symbol')['LocalTime'].idxmax().values.ravel()]
    data33 = data33.iloc[data31.groupby('Symbol')['LocalTime'].idxmax().values.ravel()]

    # join with instrument details
    data31 = data31.join(securities, on='Symbol')
    data32 = data32.join(securities, on='Symbol')
    data33 = data33.join(securities, on='Symbol')

    #separate into options and etf
    a50etf31 = data31[(data31['Type'] == 'Equity')].reset_index(drop=True) 
    a50etf32 = data32[(data32['Type'] == 'Equity')].reset_index(drop=True) 
    a50etf33 = data33[(data33['Type'] == 'Equity')].reset_index(drop=True) 

    data31 = data31[(data31['Type'] == 'Option')].reset_index(drop=True) 
    data32 = data32[(data32['Type'] == 'Option')].reset_index(drop=True) 
    data33 = data33[(data33['Type'] == 'Option')].reset_index(drop=True) 
    
    #calculate implied vol

    for index, row in data31.iterrows():
        if 'C' == row['OptionType']:
            ImpVolCallBid = ImpliedVolatilityCall(a50etf31['Bid1'], row['Strike'], r,  data31['LocalTime'], data31['Bid1'],q)


    #save results to \31.csv", \32.csv", and \33.csv"


    #create graphs

    

main()





