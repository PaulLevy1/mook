import matplotlib.pyplot as plt
import numpy

def get_price(x):
    z = x.split('"')
    # print(z)
    price = z[23]
    return price

import datetime
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
def get_historic_stonk(ticker,day1,day2):

    yf.pdr_override()
    # Tickers list
    # We can add and delete any ticker from the list to get desired ticker live data
    ticker_list = ['SPY']
    today = datetime.date.today()
    # We can get data by our choice by giving days bracket
    start_date = day1# datetime.datetime.strptime(day1,'%m/%d/%Y').date()
    end_date = day2 #datetime.datetime.strptime(day2,'%m/%d/%Y').date()
    files = []

    def getData(ticker):
        print(ticker)

    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    return data
    dataname = ticker +'_'+str(start_date) + ' - ' + str(end_date)
    #files.append(dataname)
  #  SaveData(data, dataname)

    # Create a data folder in your current dir.
    def SaveData(df, filename):
        df.to_csv('/Users/paullevy/Desktop/Test'+filename +'.csv')
        # This loop will iterate over ticker list, will pass one ticker to get data, and save that data as file.
        for tik in ticker_list:
            getData(tik)
        for i in range(0, 1):
            df1 = pd.read_csv('/Users/paullevy/Desktop/Test'+ str(files[i]) +'.csv')
            print(df1.head())

    #SaveData(data, dataname)


import urllib.request
import time

def get_historical_Eth(days):
    import csv
    import webbrowser
    # webbrowser.open('https://etherscan.io/chart/etherprice?output=csv')
    directory = '/Users/paullevy/Downloads'

    with open(directory + '/export-EtherPrice.csv', newline='\n') as f:
        reader = csv.reader(f)
        data = list(reader)
        newData = data[-days:]
    return newData


def binarySearch(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:

        mid = (high + low) // 2

        # Check if x is present at mid
        debug = arr[mid]
        if arr[mid] < x:
            low = mid + 1

        # If x is greater, ignore left half
        elif arr[mid] > x:
            high = mid - 1

        # If x is smaller, ignore right half
        else:
            return mid

            # If we reach here, then the element was not present
    return -1



def get_historical_Eth2(day1,day2):
    import csv
    #day1 = datetime.datetime.strptime(day1,'%m/%d/%Y').date()
   # day2 = datetime.datetime.strptime(day2,'%m/%d/%Y').date()

    import webbrowser
    webbrowser.open('https://etherscan.io/chart/etherprice?output=csv')
    directory = '/Users/paullevy/Downloads'

    with open(directory + '/export-EtherPrice.csv', newline='\n') as f:
        reader = csv.reader(f)
        data = list(reader)
        #newData = data[-days:]
        onlyDates = []
        for datum in data:
            onlyDates.append(datum[0])
        dates_list = [datetime.datetime.strptime(date, '%m/%d/%Y').date() for date in onlyDates[1:]]
        day1Pos = binarySearch(dates_list,day1) + 1
        day2Pos = binarySearch(dates_list, day2) + 1
        if day1Pos == 0 or day2Pos == 0:
            print('bad dates' + day1 +', '+ day2)
            raise ValueError
        newData = data[day1Pos:day2Pos]
    return newData

def disectEthHistory(ethHistory):
    dates = []
    prices = []
    for dqp in ethHistory:
        dates.append(datetime.datetime.strptime(dqp[0],'%m/%d/%Y').date())
        #dates_list = [datetime.datetime.strptime(date, '%m/%d/%Y').date() for date in onlyDates[1:]]
        prices.append(float(dqp[2]))
    datePrice = [dates,prices]
    return datePrice

def expMvAvg(alpha,prices):
    ema = []
    ema.append(prices[0])
    for i in range(len(prices)):
        if i != 0:
            temp = ema[i - 1]*(1-alpha) + alpha*prices[i]
            ema.append(temp)
    return ema

def macd_calc(prices,a = 12,b = 26,c = 9):
    alphacalc = lambda n : 2/(n+1)
    emaA = expMvAvg(alphacalc(a),prices)
    emaB = expMvAvg(alphacalc(b), prices)

   # plt.figure()
   # plt.plot(prices)
   # plt.plot(emaA)    #---- For debugging
   # plt.plot(emaB)
   # plt.show()


    macd = [] #emaA - emaB
    zippy = zip(emaA,emaB)
    for x,y in zippy:
        macd.append(x-y)
    emaC = expMvAvg(alphacalc(c), macd)
    return macd,emaC



def rsi_calc(prices,n = 14):
    def getUandD(prices):
        difference = numpy.diff(prices)
        u = []
        d = []
        for day in difference:
            if day > 0:
                u.append(day)
                d.append(0)
            elif day < 0:
                u.append(0)
                d.append(-day)
            else:
                u.append(0)
                d.append(0)
        temp = [u,d]
        return temp

    alphacalc = lambda n : 1/n
    temp = getUandD(prices)
    u = temp[0]
    d = temp[1]
    smmaU = expMvAvg(alphacalc(n),u)
    smmaD = expMvAvg(alphacalc(n),d)
    rsi = [] #smmaU/smmaD
    zippy = zip(smmaU, smmaD)
    for x, y in zippy:
        #rs.append(x/y)
        rs = x/y #figure out first case and y = 0 case
        rsi.append(100 - (100/(1+rs)))
    return rsi

class Crossover:
    "Indicates a crossover between the MACD and Signal Line"
    def __init__(self,date,direction):
        self.date = date
        self.direction = direction

    #def __lt__(self, other):
       # return self.date < other.date

def getHistoricalMacdCrosses(macd_data,dates,index):
    dates = dates[index:]
    macdHist = []
    macd = macd_data[0]
    macd = macd[index:]
    sl = macd_data[1]
    sl = sl[index:]
    [macdHist.append(macd[i] - sl[i]) for i in range(len(macd))]
    crossList = []
    for i in range(len(macdHist) - 1):
        j = i + 1
        if macdHist[j] < 0 and macdHist[j-1] > 0: # sell
            crossList.append(Crossover(dates[j],'sell'))
        elif macdHist[j] > 0 and macdHist[j-1] < 0: # buy
            crossList.append(Crossover(dates[j],'buy'))
    fig = plt.figure()
    plt.plot(dates,numpy.zeros(len(macdHist)))
    plt.plot(dates,macdHist)
    fig.autofmt_xdate()
    plt.title('MACD Histogram')
    plt.show()
    return crossList

def getHistoricalRsiStake(rsi_data,dates,halfIndex):
    rsiRiskPercentList = []
    rsiCrossList = []
    rsiMid = 50
    rsiHigh = 70
    rsiLow = 30
    maxStake = 1
    minStake = 0
    mSell = (maxStake-minStake)/(rsiHigh-rsiMid)
    mBuy = (minStake-maxStake)/(rsiMid-rsiLow)
    for i in range(len(rsi_data)):
        rsiPoint = rsi_data[i]
        if(i == 0):
            rsiRiskPercentList.append(minStake)
            continue
        if(rsiPoint > 50 and rsiPoint < 70):
            rsiRiskPercentList.append(mSell*(rsiPoint-rsiMid) + minStake)
        elif(rsiPoint >= 70):
            rsiRiskPercentList.append(maxStake)
            if rsi_data[i-1] < 70 and i > halfIndex - 1:
                rsiCrossList.append(Crossover(dates[i],'sell'))
        elif(rsiPoint < 50 and rsiPoint > 30):
            rsiRiskPercentList.append(mBuy*(rsiPoint-30) + maxStake)
        elif(rsiPoint <= 30):
            rsiRiskPercentList.append(maxStake)
            if rsi_data[i-1] > 30 and i > halfIndex - 1:
                rsiCrossList.append(Crossover(dates[i],'buy'))
        else:
            print(rsiPoint)
            raise ValueError

    return rsiRiskPercentList,rsiCrossList

def getDayAvgStock(stonkHistory):
    stonkOpenSeries = stonkHistory['Open']
    stonkCloseSeries = stonkHistory['Close']
    stonkAvg = list((stonkCloseSeries + stonkOpenSeries) / 2)
    return stonkAvg

class Transaction:
    def __init__(self,date,price,amountUSD):
        self.date = date
        self.price = price
        self.amountUSD = amountUSD

class HistoricalSimulation:
    "Simulates a RSI + MACD trading strategy"
    def __init__(self,ticker,day1,startDay,day2,stake,sellBool):
        self.ticker = ticker
        self.day1 = datetime.datetime.strptime(day1,'%m/%d/%Y').date()
        self.startDay = datetime.datetime.strptime(startDay,'%m/%d/%Y').date()
        self.day2 = datetime.datetime.strptime(day2,'%m/%d/%Y').date()
        self.stake = stake
        self.cashBalance = stake
        self.equity = 0
        self.deposited = stake
        self.buyList = []
        self.sellList = []
        self.sellBool = sellBool

    def deposit(self):
        self.deposited = self.deposited + self.stake
        self.cashBalance = self.cashBalance + self.stake

    def buy(self,date,price,ratioRsi,ratioMacd):
        if self.cashBalance < self.stake:
            self.deposit()
        if self.cashBalance >= self.stake:
            boughtAmtUSD = self.stake*(ratioRsi + ratioMacd)/2
            self.cashBalance = self.cashBalance - boughtAmtUSD
            self.equity = self.equity + boughtAmtUSD/price
            self.buyList.append(Transaction(date,price,boughtAmtUSD))
        else:
            print('BROKE BOY (buy)')
    def sell(self,date,price,ratioRsi,ratioMacd):
        if self.equity > 0:
            soldAmt = self.equity * (ratioRsi+ratioMacd)/2
            soldAmtUSD = soldAmt*price
            self.cashBalance = self.cashBalance + soldAmtUSD
            self.equity = self.equity - soldAmt
            self.sellList.append(Transaction(date,price,soldAmtUSD))
        else:
            print('BROKE BOY (sell) ')

    def calcRoi(self,endEquityBalance):
        return (self.cashBalance + endEquityBalance - self.deposited)/self.deposited



    def simulate(self):
        if self.ticker == 'ETH':
            history = get_historical_Eth2(self.day1, self.day2)
            datesAndPrice = disectEthHistory(history)
            dates = datesAndPrice[0]
            halfDatesIndex = binarySearch(dates,self.startDay)
            prices = datesAndPrice[1]
            macd_data = macd_calc(prices)
            rsi_data = rsi_calc(prices)
        else:
            history = get_historic_stonk(self.ticker,self.day1,self.day2)
            dates = list(history['Open'].index)
            halfDatesIndex = binarySearch(dates,self.startDay)
            while halfDatesIndex == -1:
                self.startDay = self.startDay + datetime.timedelta(days=2)
                halfDatesIndex = binarySearch(dates,self.startDay) #trading days
            prices = getDayAvgStock(history)
            macd_data = macd_calc(prices)
            rsi_data = rsi_calc(prices)

        macdCrossList = getHistoricalMacdCrosses(macd_data, dates,halfDatesIndex)
        rsiTuple = getHistoricalRsiStake(rsi_data, dates[1:],halfDatesIndex)
        startPrice = prices[halfDatesIndex]
        endPrice = prices[-1]
        returnOverTime = (endPrice-startPrice)/startPrice*100
        rsiStakeList = rsiTuple[0]
        rsiCrossList = rsiTuple[1]

        fullCrossList = []
        [fullCrossList.append(cross) for cross in macdCrossList]
        [fullCrossList.append(cross) for cross in rsiCrossList]
        fullCrossList = sorted(fullCrossList,key=lambda cross : cross.date)
        r = range(1,len(fullCrossList))
        i = r.start
        while i < r.stop and i >= r.start:
            if fullCrossList[i].date == fullCrossList[i-1].date:
                del fullCrossList[i]
                r = range(1,len(fullCrossList))
                i = i - 1
            i = i + r.step

        plt.figure()
        plt.plot(dates[1:], rsiStakeList)
        plt.title('Rsi staking')
        plt.show()

        time = dates
        plt.figure()
        plt.plot(time, prices)
        plt.title('Price vs. Time')
        plt.xlabel('Date')
        plt.ylabel(self.ticker +' Price ($)')

        f1 = plt.figure()
        realMacd, = plt.plot(time, macd_data[0], label='MACD')
        signalLine, = plt.plot(time, macd_data[1], label='Signal Line')
        zeroLine, = plt.plot(time, numpy.zeros(len(prices)), label='Zero Line')
        f1.autofmt_xdate()
        plt.title('MACD')
        plt.legend(handles=[realMacd, signalLine, zeroLine])

        f2 = plt.figure()
        plt.plot(time[1:], rsi_data)
        f2.autofmt_xdate()
        # plt.plot(numpy.zeros(len(ethPrices)))
        zeros = numpy.array(numpy.zeros(len(prices) - 1))
        plusn = lambda x, n: x + n

        fifty = plusn(zeros, 50)
        seventy = plusn(zeros, 70)
        thirty = plusn(zeros, 30)
        plt.plot(time[1:], fifty)
        plt.plot(time[1:], seventy)
        plt.plot(time[1:], thirty)
        # plt.ylim(ymax=101,ymin=-1)
        plt.title('RSI')
        plt.show()

        prices_ts = pd.Series(prices,dates)
        rsiStake_ts = pd.Series(rsiStakeList,dates[1:])



        for cross in fullCrossList:
            if cross.direction =='buy':
                rsi_stake = rsiStake_ts[cross.date] # [0% to 100%]
                price = prices_ts[cross.date]
                self.buy(cross.date,price,rsi_stake,1)
            elif cross.direction =='sell':
                rsi_stake = rsiStake_ts[cross.date]  # [0% to 100%]
                price = prices_ts[cross.date]
                if self.sellBool == 1:
                    self.sell(cross.date,price,rsi_stake,1)
            else:
                raise ValueError

        print('ticker: ' + self.ticker + '\n')
        print('day1: ' + str(self.startDay) + '\n')
        print('day2: ' + str(self.day2) + '\n')
        print('cash: ' + str(self.cashBalance) + '\n')
        lastDate = self.day2 - datetime.timedelta(days=1)
        equityBalance = self.equity*prices_ts[lastDate]
        print('equity in $: ' + str(equityBalance))
        roi = self.calcRoi(equityBalance)
        print('roi: ' + str(roi))

        fig = plt.figure()
        plt.plot(time, prices)
        fig.autofmt_xdate()
        plt.title('Price vs. Time for '+ self.ticker + ' with buy(green) and sell(red)')
        plt.xlabel('Date')
        plt.ylabel(self.ticker + ' Price ($)')
        for buy in self.buyList:
            plt.plot(buy.date,buy.price,'g*')
        for sell in self.sellList:
            plt.plot(sell.date,sell.price,'rx')


        plt.legend('buy','sell')
        plt.show()
        roi = roi * 100
        r = [roi,returnOverTime]
        return r

#test = HistoricalSimulation('ETH','1/1/2019','1/1/2020',1000)
#test.simulate()
#stonkHistory = get_historic_stonk('SPY','1/1/2019','1/1/2020')
#stonkOpenSeries = stonkHistory['Open']
#stonkCloseSeries = stonkHistory['Close']
#stonkDates = list(stonkOpenSeries.index)
#stonkAvg = list((stonkCloseSeries + stonkOpenSeries)/2)
#stonkMACD = macd_calc(stonkOpenSeries)
#stonkRSI = rsi_calc(stonkOpenSeries)

# ethHistory = get_historical_Eth(90)
#ethHistory = get_historical_Eth2('1/1/2020','7/6/2020')
#print(ethHistory)
#ethDateAndPrice = disectEthHistory(ethHistory)
#ethDates = ethDateAndPrice[0]
#ethPrices = ethDateAndPrice[1]
#ethMACD_Data = macd_calc(ethPrices)
#ethRSI = rsi_calc(ethPrices)
#macdCrossList = getHistoricalMacdCrosses(ethMACD_Data,ethDates)
#rsiStakeList = getHistoricalRsiStake(ethRSI,ethDates[1:])
#plt.figure()
#plt.plot(ethDates[1:],rsiStakeList)
#plt.title('Rsi staking')
#plt.show()

#time = ethDates
#plt.figure()
#plt.plot(time,ethPrices)
#plt.title('Ethereum Price vs. Time')
#plt.xlabel('Date')
#plt.ylabel('ETH Price ($)')


#plt.figure()
#realMacd, = plt.plot(time,ethMACD_Data[0],label='MACD')
#signalLine, = plt.plot(time,ethMACD_Data[1],label='Signal Line')
#zeroLine, = plt.plot(time,numpy.zeros(len(ethPrices)),label='Zero Line')
#plt.title('Ethereum MACD')
#plt.legend(handles=[realMacd,signalLine,zeroLine])

#plt.figure()
#plt.plot(time[1:],ethRSI)
# plt.plot(numpy.zeros(len(ethPrices)))
#zeros = numpy.array(numpy.zeros(len(ethPrices) - 1))
#plusn = lambda x,n:x+n

#fifty = plusn(zeros,50)
#seventy = plusn(zeros,70)
#thirty = plusn(zeros,30)
#plt.plot(time[1:],fifty)
#plt.plot(time[1:],seventy)
#plt.plot(time[1:],thirty)
#plt.ylim(ymax=101,ymin=-1)
#plt.title('Ethereum RSI')
#plt.show()
#url = "https://api.nomics.com/v1/currencies/ticker?key=7e97be238574825871fdd0ec8bcc5e69&ids=ETH&interval=1d"
#url2 = 'https://api.etherscan.io/api?module=stats&action=ethprice&apikey=RAR6HFCEVMUZXJ1G4VYD7I4ZR5M2KSB8Y7'







#for a in range(1):
    #   print('\n')
    #x = urllib.request.urlopen(url).read()
    ## print(x)
    # x = x.decode("utf-8")
    #price = get_price(x)
    #print(price)
    #time.sleep(5)



