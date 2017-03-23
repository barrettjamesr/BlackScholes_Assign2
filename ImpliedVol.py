import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def FullBlackScholesCall(S, X, r, t, vol, q):
    d1 = (np.log(S/X)+(r-q+vol**2 / 2)*t)/(vol*t**0.5)
    d2 = d1 - vol * t ** 0.5
    
    return S / np.exp(q * t) * norm.cdf(d1) - norm.cdf(d2) * X / np.exp(r * t)

def FullBlackScholesPut(S, X, r, t, vol, q):
    d1 = (np.log(S/X)+(r-q+vol**2 / 2)*t)/(vol*t**0.5)
    d2 = d1 - vol * t ** 0.5
    
    return norm.cdf(0-d2) * X / np.exp(r * t) -S / np.exp(q * t) * norm.cdf(0-d1)

def Vega(S, X, r, t, vol, q):
    d1 = (np.log(S/X)+(r-q+vol**2 / 2)*t)/(vol*t**0.5)

    return S / np.exp(q * t) * t ** 0.5 * norm.cdf(d1)

def ImpliedVolatilityCall(S, X, r, t, C_true, q, sigmaHat):
    tol = 1e-8
    sigmadiff = 1.0
    intI = 1
    iMax = 100

    sigma = sigmaHat

    while (sigmadiff >= tol and intI < iMax):
        if 0 == sigma:
            return 'NaN'
        C = FullBlackScholesCall(S, X, r, t, sigma, q)
        Cvega = Vega (S, X, r, t, sigma, q)
        if 0 == round(Cvega,10):
            return 'NaN'
        increment = (C - C_true)/Cvega
        sigma = sigma - increment
        intI=intI+1
        sigmadiff = abs(increment)

    return sigma

def ImpliedVolatilityPut(S, X, r, t, P_true, q, sigmaHat):
    tol = 1e-8
    sigmadiff = 1.0
    intI = 1
    iMax = 100

    sigma = sigmaHat

    while (sigmadiff >= tol and intI < iMax):
        if 0 == sigma:
            return 'NaN'
        P = FullBlackScholesPut(S, X, r, t, sigma, q)
        Pvega = Vega (S, X, r, t, sigma, q)
        if 0 == round(Pvega,10):
            return 'NaN'
        increment = (P - P_true)/Pvega
        sigma = sigma - increment
        intI=intI+1
        sigmadiff = abs(increment)

    return sigma

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
    impliedVolCall = pd.DataFrame(columns=('Strike', 'BidVolC', 'AskVolC'))
    impliedVolPut = pd.DataFrame(columns=('Strike', 'BidVolP', 'AskVolP'))

    for index, row in data31.iterrows():
        t = float((row['Expiry'].date() -row['LocalTime'].date()).days)/365
        spot = (float(a50etf31['Bid1'])+float(a50etf31['Ask1'])) /2 
        sigmaHat = np.sqrt(2*abs((np.log(spot/float(row['Strike'])) + (r-q)*t)/t ))

        if 'C' == row['OptionType']:
            ImpVolCallBid = ImpliedVolatilityCall(spot, float(row['Strike']), r, t, float(row['Bid1']),q, sigmaHat)
            ImpVolCallAsk = ImpliedVolatilityCall(spot, float(row['Strike']), r, t, float(row['Ask1']),q, sigmaHat)

            series = pd.Series([row['Strike'], ImpVolCallBid, ImpVolCallAsk], index=['Strike', 'BidVolC', 'AskVolC'])
            impliedVolCall.loc[float(row['Strike'])] = series

        if 'P' == row['OptionType']:
            ImpVolPutBid = ImpliedVolatilityPut(spot, float(row['Strike']), r, t, float(row['Bid1']),q, sigmaHat)
            ImpVolPutAsk = ImpliedVolatilityPut(spot, float(row['Strike']), r, t, float(row['Ask1']),q, sigmaHat)

            series = pd.Series([row['Strike'], ImpVolPutBid, ImpVolPutAsk], index=['Strike', 'BidVolP', 'AskVolP'])
            impliedVolPut.loc[float(row['Strike'])] = series

    impliedVol31 = impliedVolPut.join(impliedVolCall.set_index('Strike'), on='Strike', how='left').sort_values('Strike')

    impliedVolCall = pd.DataFrame(columns=('Strike', 'BidVolC', 'AskVolC'))
    impliedVolPut = pd.DataFrame(columns=('Strike', 'BidVolP', 'AskVolP'))

    for index, row in data32.iterrows():
        t = float((row['Expiry'].date() -row['LocalTime'].date()).days)/365
        spot = (float(a50etf32['Bid1'])+float(a50etf32['Ask1'])) /2 
        sigmaHat = np.sqrt(2*abs((np.log(spot/float(row['Strike'])) + (r-q)*t)/t ))

        if 'C' == row['OptionType']:
            ImpVolCallBid = ImpliedVolatilityCall(spot, float(row['Strike']), r, t, float(row['Bid1']),q, sigmaHat)
            ImpVolCallAsk = ImpliedVolatilityCall(spot, float(row['Strike']), r, t, float(row['Ask1']),q, sigmaHat)

            series = pd.Series([row['Strike'], ImpVolCallBid, ImpVolCallAsk], index=['Strike', 'BidVolC', 'AskVolC'])
            impliedVolCall.loc[float(row['Strike'])] = series

        if 'P' == row['OptionType']:
            ImpVolPutBid = ImpliedVolatilityPut(spot, float(row['Strike']), r, t, float(row['Bid1']),q, sigmaHat)
            ImpVolPutAsk = ImpliedVolatilityPut(spot, float(row['Strike']), r, t, float(row['Ask1']),q, sigmaHat)

            series = pd.Series([row['Strike'], ImpVolPutBid, ImpVolPutAsk], index=['Strike', 'BidVolP', 'AskVolP'])
            impliedVolPut.loc[float(row['Strike'])] = series

    impliedVol32 = impliedVolPut.join(impliedVolCall.set_index('Strike'), on='Strike', how='left').sort_values('Strike')

    impliedVolCall = pd.DataFrame(columns=('Strike', 'BidVolC', 'AskVolC'))
    impliedVolPut = pd.DataFrame(columns=('Strike', 'BidVolP', 'AskVolP'))

    for index, row in data33.iterrows():
        t = float((row['Expiry'].date() -row['LocalTime'].date()).days)/365
        spot = (float(a50etf33['Bid1'])+float(a50etf33['Ask1'])) /2 
        sigmaHat = np.sqrt(2*abs((np.log(spot/float(row['Strike'])) + (r-q)*t)/t ))

        if 'C' == row['OptionType']:
            ImpVolCallBid = ImpliedVolatilityCall(spot, float(row['Strike']), r, t, float(row['Bid1']),q, sigmaHat)
            ImpVolCallAsk = ImpliedVolatilityCall(spot, float(row['Strike']), r, t, float(row['Ask1']),q, sigmaHat)

            series = pd.Series([row['Strike'], ImpVolCallBid, ImpVolCallAsk], index=['Strike', 'BidVolC', 'AskVolC'])
            impliedVolCall.loc[float(row['Strike'])] = series

        if 'P' == row['OptionType']:

            ImpVolPutBid = ImpliedVolatilityPut(spot, float(row['Strike']), r, t, float(row['Bid1']),q, sigmaHat)
            ImpVolPutAsk = ImpliedVolatilityPut(spot, float(row['Strike']), r, t, float(row['Ask1']),q, sigmaHat)

            series = pd.Series([row['Strike'], ImpVolPutBid, ImpVolPutAsk], index=['Strike', 'BidVolP', 'AskVolP'])
            impliedVolPut.loc[float(row['Strike'])] = series

    impliedVol33 = impliedVolPut.join(impliedVolCall.set_index('Strike'), on='Strike', how='left').sort_values('Strike')


    #save results to \31.csv", \32.csv", and \33.csv"
    impliedVol31.to_csv('C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\31.csv', index=False)
    impliedVol32.to_csv('C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\32.csv', index=False)
    impliedVol33.to_csv('C:\\Users\\James\\SkyDrive\\Documents\\HKU\\TechniquesInCompFin\\33.csv', index=False)

    #create graphs
    plt.figure(); impliedVol31.plot();

    #check for arbitrage?




main()

#sigmaHat = np.sqrt(2*abs( (np.log(100/120) + (0.01)*0.5)/0.5 ))
#print(ImpliedVolatilityCall(100, 120, 0.01, 0.5, 0.774139, 0, sigmaHat))
#print(ImpliedVolatilityPut(100, 120, 0.01, 0.5, 20.175636, 0, sigmaHat))

