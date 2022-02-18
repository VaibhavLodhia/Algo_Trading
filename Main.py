# ----------Login-------
import http.client
from unittest import result
import pandas as pd
import socket
import mimetypes
# from stockdata import store_prices
from datetime import datetime
import getmac
import threading
import websockets
import json
import itertools
from runindicators import MACD, Rsi, Stochactic_Oscillator, bollinger_dataframe, database, moving_avg, movingavg_slope
from createdatbase import store_prices


#Reading data from excel
df = pd.read_excel('D:/Emergeflow/NSE/Configurations/Config.xlsx')
price_list=[]
dat ={}
#Giving keys from first column and values from second column to the dict 
dat.update(df.set_index('Name').to_dict())
del dat['Description']
Cred = {}
Cred.update(dat['Value'])

# Acessing values from dictonary and oassing them to variables
Client_code = Cred.get('Client_code')
Password  = Cred.get('Password')
Api_key = Cred.get('Api_key')
Stock_name = Cred.get('Stock_name')
Symbol_token = str(Cred.get('Symbol_token'))
From_date = Cred.get('From_date').strftime('%Y-%m-%d')
To_date = Cred.get('To_date').strftime('%Y-%m-%d')
Start_time = str(Cred.get('Start_time').strftime('%H:%M'))
End_time= str(Cred.get('End_time').strftime('%H:%M'))
Interval = Cred.get('Interval')
Indicat = Cred.get('Indicators')
Create_datbase_only = Cred.get('Create_datbase_only')
Indicators = list(Indicat.split(','))
# Indicators.append(Indicat)

#Getting IP and Mac Address
hostname = socket.gethostname()
Ipad= socket.gethostbyname(hostname)

# ---------Login to smartapi---------

conn = http.client.HTTPSConnection(
    "apiconnect.angelbroking.com"
    )
payload = "{\n\"clientcode\":\""+Client_code+"\",\n\"password\":\""+Password+"\"\n}"
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-UserType': 'USER',
    'X-SourceID': 'WEB',
    'X-ClientLocalIP': 'CLIENT_LOCAL_IP',
    'X-ClientPublicIP': 'CLIENT_PUBLIC_IP',
    'X-MACAddress': 'MAC_ADDRESS',
    'X-PrivateKey': Api_key
  }
conn.request(
    "POST", 
    "/rest/auth/angelbroking/user/v1/loginByPassword",
     payload,
     headers)

res = conn.getresponse()
data = res.read()
# print(data.decode("utf-8"))
dict = json.loads(data.decode("utf-8"))
tok = dict.get('data')
jwt = tok.get('jwtToken')

# -----------Getting historical data----


import http.client
token = 'Bearer {}'.format(jwt)
elements_data =[]
conn = http.client.HTTPSConnection("apiconnect.angelbroking.com")
payload = "{\r\n\"exchange\":\""+Stock_name+"\",\r\n\"symboltoken\":\""+Symbol_token+"\",\r\n\"interval\":\""+Interval+"\",\r\n\"fromdate\":\""+From_date+" "+Start_time+"\",\r\n \"todate\":\""+To_date+" "+End_time+"\"\r\n}"
headers = {
  'X-PrivateKey': Api_key,
  'Accept': 'application/json',
  'X-SourceID': 'WEB',
  'X-ClientLocalIP': Ipad,
  'X-ClientPublicIP': 'CLIENT_PUBLIC_IP',
  'X-MACAddress': getmac.get_mac_address(),
  'X-UserType': 'USER',
  'Authorization': token,
  'Accept': 'application/json',
  'X-SourceID': 'WEB',
  'Content-Type': 'application/json'
}



conn.request("POST", "/rest/secure/angelbroking/historical/v1/getCandleData", payload, headers)
res = conn.getresponse()
data = res.read()

#Storing the historical data into dictonary
stock_dict = json.loads(data.decode("utf-8"))
# print(data.decode("utf-8"))
# Pushing all the historical prices to Price list
price_list.append(stock_dict.get('data'))
dat = price_list[0]
results =[]

# Checking gain or loss by the logic:-'gain' if close>open in that row. Value should be 'loss' if open>close.
for i in dat:
      results.extend(i[0].split('T'))
      results.extend(i[1:])
      if i[1:][0]>i[1:][3]:
            results.append("loss")                 
      if i[1:][0]<i[1:][3]:            
            results.append("gain")
      if i[1:][0]==i[1:][3]:
            results.append("safe")

# Generating sublist for each date and time
op = [results[i:i+8] for i in range(0,len(results),8)]
# print(op)
# Pushing each sublist with 8 entries into database

# database()

# threading.Timer()
def Indict():
      for i in Indicators:
            if i == 'Moving Average':
                  moving_avg()

            if i  == 'RSI':
                  Rsi()
            
            if i == 'Bollinnger Bands':
                  bollinger_dataframe()
            
            if i == "MACD":
                  MACD()

            if i == 'Stochastic Oscillator':
                  Stochactic_Oscillator()

            if i == 'Moving average slope':
                  movingavg_slope()

def callfunction():
      database()
      Indict()

if Create_datbase_only == 'yes':
      for entry in op:
          if len(entry) == 8:
            store_prices(entry)
elif Create_datbase_only == 'no':
      for entry in op:
          if len(entry) == 8:
            store_prices(entry)
      threading.Timer(600,callfunction())