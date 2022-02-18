import sqlite3
from statistics import fmean, stdev
from tkinter.tix import Select
from turtle import width
from unittest import result
# from matplotlib.lines import _LineStyle
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import os
import pandas_ta as ta
import matplotlib.pyplot as plt
from operator import itemgetter
con = sqlite3.connect('prices.db')
c = con.cursor()


def database():
    df1 = pd.read_excel('Configurations/Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')
    Stock = Cred.get('Stock_name')
    # print(Stock)
# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    c.execute('SELECT * FROM {}'.format(Stock))
    # print(c.fetchall())
    col = ['Id','Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close', 'Stock_volume', 'Stock_Results']
    df = pd.DataFrame(c.fetchall(), columns=col)
    df.to_excel("database.xlsx")


#Generating time frame
def timeframe():
    df1 = pd.read_excel('Configurations/Timeframe_Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')
    Frame = Cred.get('Time_frame')
    tf_source = Cred.get('Time_frame_Source')


    dat = []
    log=[]
    open =[]
    close = []
    cl =[]
    low = []
    lowval =[]
    high = []
    hg=[]
    volume = []

    # Fetching data from database 
    c.execute('SELECT * FROM {} '.format(Stock))

    # Appending data to list 
    dat.append(c.fetchall())
    data = dat[0]
    
    log.append([data[i:i+Frame] for i in range(0,len(data),Frame)])

    # Appending data to thier equivalent lists
    for i in range(len(log[0])):
        for j in log[0][i]:
            open.append(j[3])
            high.append(j[4])
            low.append(j[5])
            close.append(j[6])
            volume.append(j[7])
    
    # Grouping  each list acording to frame
    cl.append([close[i:i+Frame] for i in range(0,len(close),Frame)])
    hg.append([high[i:i+Frame] for i in range(0,len(high),Frame)])
    lowval.append([low[i:i+Frame] for i in range(0,len(low),Frame)])
    li =[]
    lo= []
    
    # Getting high, low, close, open values from timeframe 
    for i in range(len(hg[0])):
        li.append(sorted(hg[0][i], reverse=True)[0])

    for i in range(len(lowval[0])):
        lo.append(sorted(lowval[0][i])[0])
    Stock_open = open[::Frame] 
    Stock_high= li
    Stock_low= lo
    Stock_close= list(map(itemgetter(-1),cl[0])) 
    if tf_source =='Stock_open':
        print(Stock_open)      
    elif tf_source == 'Stock_low':
        print(Stock_low)   
    elif tf_source == 'Stock_high':
        print(Stock_high)     
    elif tf_source == 'Stock_close':
        print(Stock_close)     

# defining moving average function

def moving_avg():
    df1 = pd.read_excel('Configurations/Movingavg_Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')
    Length = Cred.get('EMA_SMA_Length')
    mov = Length.split(',')
    moving = [int(i) for i in mov]
  
    # Fetching data from database 
    c.execute('SELECT * FROM {} '.format(Stock))

    col = ['Id','Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close', 'Stock_volume', 'Stock_Results']
    df = pd.DataFrame(c.fetchall(), columns=col)
    
    # df.ta.sma(close='Stock_Close', length=5, append=True)
    df.ta.ema(close='Stock_Close', length=moving[0], append=True)
    df.ta.sma(close='Stock_Close', length=moving[1], append=True)

    if not os.path.exists('Movingavg_results'):
        os.makedirs('Movingavg_results')


    df.to_excel("Movingavg_results/movingavg.xlsx")
    datee=[]
    datelist = df['Stock_Time'].tolist()
    for i in datelist:
        datee.append(i.removesuffix(':00+05:30'))
    out = []
    cl = []
    # Closelist = df['Stock_Close'].tolist()
    dout = []
    ema = df['EMA_10'].tolist()
    sma = df['SMA_20'].tolist()
    kout =[]
    out.append([datee[i:i+50] for i in range(0,len(datee),50)])
    dout.append([ema[i:i+50] for i in range(0,len(ema),50)])
    kout.append([sma[i:i+50] for i in range(0,len(sma),50)])
    # cl.append([Closelist[i:i+50] for i in range(0,len(Closelist),50)])

    # if not os.path.exists('Movingavg_graphs'):
    #     os.makedirs('Movingavg_graphs')

    for j in range(len(out[0])):
            # print(da)
        x = out[0][j]
        y=  dout[0][j]
        y1 = kout[0][j]
        # y2 = cl[0][j]
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.plot(x, y, label='ema', color = '#EBD2BE')
        # plt.plot(x, y2, label='CLOSE', color = '#0f0f0f')
        plt.plot(x, y1, label='sma', color='#E5A4CB')
        plt.xticks(rotation=90)
        plt.savefig('Movingavg_results/op{}.jpg'.format(j))

def movingavg_slope():

    moving_avg()
    result = []
    graphresult = []
    out = []
    df = pd.read_excel('Movingavg_results/Moving_Average.xlsx')
    avglist = df['Stock_movingavg'].tolist()
    timelist = df['Stock_time'].tolist()
    datelist = df['Stock_date'].tolist()
    print(len(avglist))
    for i in range(len(avglist)-1):
        result.append("{:.2f}".format(avglist[i+1] - avglist[i]))
    #     # for j in range(1,len(avglist)-1):
        #     result.append(avglist[j] - avglist[i])
    out.append([timelist[i:i+35] for i in range(0,len(timelist),35)])
    graphresult.append([result[i:i+35] for i in range(0,len(result),35)])

    output= pd.DataFrame({'Stock_Date':datelist[:len(result)], 'Stock_time': timelist[:len(result)], 'Stock_movingavg':avglist[:len(result)], 'Moving_avgslope': result})
    if not os.path.exists('Movingavg_sloperesults'):
        os.makedirs('Movingavg_sloperesults')
    output.to_excel("Movingavg_sloperesults/Movingavg_slope.xlsx")
    # print(len(datelist[:len(result)]))
    # if not os.path.exists('Movingavg_slopegraphs'):
    #     os.makedirs('Movingavg_slopegraphs')

    for j in range(len(out[0])):
            # print(da)
        x = out[0][j]
        y=  graphresult[0][j]
        # y1 = kout[0][j]
        # y2 = cl[0][j]
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.plot(x, y, label='Slope', color = '#EBD2BE')
        # plt.plot(x, y2, label='CLOSE', color = '#0f0f0f')
        # plt.plot(x, y1, label='sma', color='#E5A4CB')
        plt.xticks(rotation=90)
        plt.savefig('Movingavg_sloperesults/op{}.jpg'.format(j))

    # print(result[1])

def Rsi():
    df1 = pd.read_excel('Configurations/Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')

    out = []
    clow = []
    chigh =[]
    kout =[]
    high = []
    low = []
    c.execute('SELECT * FROM {}'.format(Stock))
    # print(c.fetchall())
    col = ['Id','Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close', 'Stock_volume', 'Stock_Results']
    ticker = pd.DataFrame(c.fetchall(), columns=col)

    delta = ticker['Stock_Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down

    ticker['RSI'] = 100 - (100/(1 + rs))
    print(ticker)
    if not os.path.exists('RSI_results'):
        os.makedirs('RSI_results')
    ticker.to_excel("RSI_results/RSI.xlsx")
    # print(ticker)

    rsi =ticker['RSI'].tolist()
    for k in range(len(rsi)):
        high.append(70)
        low.append(30)
    # print(len(high))
    datee=[]
    datelist = ticker['Stock_Time'].tolist()
    for i in datelist:
        datee.append(i.removesuffix(':00+05:30'))

    out.append([datee[i:i+50] for i in range(0,len(datee),50)])
    kout.append([rsi[i:i+50] for i in range(0,len(rsi),50)])
    clow.append([low[i:i+50] for i in range(0,len(low),50)])
    chigh.append([high[i:i+50] for i in range(0,len(high),50)])

    # if not os.path.exists('RSI_graphs'):
    #     os.makedirs('RSI_graphs')

    for j in range(len(out[0])):
        x = out[0][j]
        y = clow[0][j]
        y1 = kout[0][j]
        y2 = chigh[0][j]
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.plot(x,y, color = '#0f0f0f',linestyle='--')
        plt.plot(x,y2, color = '#0f0f0f',linestyle='--' )
        plt.plot(x, y1, label='RSI', color='#E5A4CB',)
        plt.xticks(rotation=90)
        plt.savefig('RSI_results/op{}.jpg'.format(j))



def get_bollinger_bands():
    df1 = pd.read_excel('Configurations/BollingerBand_Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')
    Bollinger_source= Cred.get('Bollinger_Source')

    Bollinger_rate = Cred.get('Bollinger_rate')

    pri =[]
    data = []
    sma =[]
    std =[]
    bollinger_up =[]
    bollinger_down= []
    c.execute('SELECT {} FROM {} '.format(Bollinger_source,Stock))
    pri.append(c.fetchall())
    prices = pri[0]
    out = [item for t in prices for item in t]
    data.append([out[i:i+Bollinger_rate] for i in range(0,len(out),Bollinger_rate)])

    for i in range(len(data[0])):
        sma.append(fmean(data[0][i]))
        std.append(stdev(data[0][i]))

    for j in range(len(sma)):
        bollinger_up.append(sma[j] + std[j] *2)
        bollinger_down.append(sma[j] - std[j] *2)
    
    return bollinger_down, bollinger_up

def bollinger_dataframe():
    df1 = pd.read_excel('Configurations/BollingerBand_Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')
    Bollinger_rate = Cred.get('Bollinger_rate')
    dat =[]
    date =[]
    time = []
    close =[]
    bollinger_down, bollinger_up = get_bollinger_bands()
    c.execute('SELECT Stock_date,Stock_Time,Stock_Close FROM {} '.format(Stock))

    # Appending values to list
    dat.append(c.fetchall())
    for j in range(len(dat[0])):
        date.append(dat[0][j][0])
        time.append(dat[0][j][1])
        close.append(dat[0][j][2])
    dataframe = pd.DataFrame(list(zip(date[::Bollinger_rate], time[::Bollinger_rate], close[::Bollinger_rate], bollinger_down, bollinger_up )), columns=['Stock_date', 'Stock_time' , 'Stock_Close', 'Bollinger_down', 'Bollinger_up'])
    # print(dataframe)
    if not os.path.exists('Bollinger_results'):
        os.makedirs('Bollinger_results')

    dataframe.to_excel("Bollinger_results/Bollinger.xlsx")
    df = pd.read_excel("Bollinger_results/Bollinger.xlsx")

    Boll_d = df['Bollinger_down'].tolist()
    Boll_u = df['Bollinger_up'].tolist()
    datee=[]
    datelist = df['Stock_time'].tolist()
    for i in datelist:
        datee.append(i.removesuffix(':00+05:30'))
    out = []
    cl = []
    Closelist = df['Stock_Close'].tolist()
    dout = []
    kout =[]
    out.append([datee[i:i+50] for i in range(0,len(datee),50)])
    dout.append([Boll_d[i:i+50] for i in range(0,len(Boll_d),50)])
    kout.append([Boll_u[i:i+50] for i in range(0,len(Boll_u),50)])
    cl.append([Closelist[i:i+50] for i in range(0,len(Closelist),50)])

    # if not os.path.exists('Bollinger_graphs'):
    #     os.makedirs('Bollinger_graphs')

    for j in range(len(out[0])):
            # print(da)
        x = out[0][j]
        y=  dout[0][j]
        y1 = kout[0][j]
        y2 = cl[0][j]
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.plot(x, y, label='Bollinger down', color = '#EBD2BE')
        plt.plot(x, y2, label='CLOSE', color = '#0f0f0f')
        plt.plot(x, y1, label='Bollinger up', color='#E5A4CB')
        plt.xticks(rotation=90)
        plt.savefig('Bollinger_results/op{}.jpg'.format(j))

def MACD():
    df1 = pd.read_excel('Configurations/Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')

    datee = []
    c.execute('SELECT * FROM {}'.format(Stock))
    # print(c.fetchall())
    col = ['Id','Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close', 'Stock_volume', 'Stock_Results']
    df = pd.DataFrame(c.fetchall(), columns=col)
    # print(df)
    k = df['Stock_Close'].ewm(span=12, adjust=False).mean()

    d = df['Stock_Close'].ewm(span=26, adjust=False).mean()

    macd = k - d 

    e = macd.ewm(span=9, adjust=False).mean()
    df ['MACD'] = macd

    df ['Signal'] = e

    datamacd = df[['Id', 'Stock_Date', 'Stock_Time', 'Stock_Close', 'MACD', 'Signal']]

    if not os.path.exists('MACD_results'):
        os.makedirs('MACD_results')

    datamacd.to_excel("MACD_results/MACD.xlsx")


    macdlist = df['MACD'].tolist()
    datelist = df['Stock_Time'].tolist()

    Signalist = df['Signal'].tolist()
    
    for i in datelist:
        datee.append(i.removesuffix(':00+05:30'))
    out = []
    macdout = []
    signalout =[]
    out.append([datee[i:i+50] for i in range(0,len(datee),50)])
    macdout.append([macdlist[i:i+50] for i in range(0,len(macdlist),50)])
    signalout.append([Signalist[i:i+50] for i in range(0,len(Signalist),50)])

    # if not os.path.exists('MACD_graphs'):
    #     os.makedirs('MACD_graphs')

    for j in range(len(out[0])):
            # print(da)
        x = out[0][j]
        y=  macdout[0][j]
        y1 = signalout[0][j]
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.plot(x, y, label='AMD MACD', color = '#EBD2BE')
        plt.plot(x, y1, label='Signal Line', color='#E5A4CB')
        plt.xticks(rotation=90)
        plt.savefig('MACD_results/op{}.jpg'.format(j))

def  Stochactic_Oscillator():
    df1 = pd.read_excel('Configurations/Stochastic_Config.xlsx')
    dat ={}

#Giving keys from first column and values from second column to the dict 
    dat.update(df1.set_index('Name').to_dict())
    del dat['Description']
    Cred = dat.get('Value')

# Acessing values from dictonary and passing them to variables
    Cred.update(dat['Value'])
    Stock = Cred.get('Stock_name')
    K = Cred.get('Stochastic_Oscillator_K_period')
    D = Cred.get('Stochastic_Oscillator_d_period')
    c.execute('SELECT * FROM {}'.format(Stock))
    # print(c.fetchall())
    col = ['Id','Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close', 'Stock_volume', 'Stock_Results']
    df = pd.DataFrame(c.fetchall(), columns=col)

    df['n_high'] = df['Stock_high'].rolling(K).max()
    # Adds an "n_low" column with min value of previous 14 periods
    df['n_low'] = df['Stock_low'].rolling(K).min()
    # Uses the min/max values to calculate the %k (as a percentage)
    df['%K'] = (df['Stock_Close'] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])
    # Uses the %k to calculates a SMA over the past 3 values of %k
    df['%D'] = df['%K'].rolling(D).mean()

    print(df[['Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close','%K', '%D']])
    so = df[['Stock_Date','Stock_Time','Stock_Open', 'Stock_high', 'Stock_low', 'Stock_Close','%K', '%D']]

    if not os.path.exists('SO_results'):
        os.makedirs('SO_results')

    so.to_excel("SO_results/Stochastic.xlsx")

    K_list = df['%K'].tolist()
    D_list = df['%D'].tolist()
    
    datee=[]
    datelist = df['Stock_Time'].tolist()
    for i in datelist:
        datee.append(i.removesuffix(':00+05:30'))
    out = []
    dout = []
    kout =[]
    out.append([datee[i:i+50] for i in range(0,len(datee),50)])
    dout.append([D_list[i:i+50] for i in range(0,len(D_list),50)])
    kout.append([K_list[i:i+50] for i in range(0,len(K_list),50)])

    # if not os.path.exists('SO_graphs'):
    #     os.makedirs('SO_graphs')

    for j in range(len(out[0])):
            # print(da)
        x = out[0][j]
        y=  dout[0][j]
        y1 = kout[0][j]
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)
        plt.plot(x, y, label='Slow', color = '#EBD2BE')
        plt.plot(x, y1, label='Fast', color='#E5A4CB')
        plt.xticks(rotation=90)
        plt.savefig('SO_results/op{}.jpg'.format(j))
    
# Rsi()
# database()
# moving_avg()
# bollinger_dataframe()
# Rsi()