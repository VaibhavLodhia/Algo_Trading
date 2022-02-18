import sqlite3
import pandas as pd

# connecting databse
con = sqlite3.connect('prices.db')

#Reading data from excel
df = pd.read_excel('Configurations/Config.xlsx')
dat ={}
#Giving keys from first column and values from second column to the dict 
dat.update(df.set_index('Name').to_dict())
del dat['Description']
Cred = {}
# Acessing values from dictonary and passing them to variables
Cred.update(dat['Value'])
Stock = Cred.get('Stock_name')

c = con.cursor()

# Defining function which takes list of 8 entries 
def store_prices(elements):

    # Checking if table already exists 
    listOfTables = c.execute(''' SELECT name FROM sqlite_master WHERE type='table' AND name=? ''',(Stock,)).fetchall()
   
    # If table dosn't exist creating a new one
    if listOfTables == []:
        sql_cmd = '''CREATE TABLE {}(id integer primary key,Stock_Date,Stock_Time,Stock_Open, Stock_high, Stock_low, Stock_Close, Stock_volume, Stock_Results)'''.format(Stock)
        c.execute(sql_cmd)
        
        # Inserting entries to table
        values = "INSERT INTO {}(Stock_Date,Stock_Time,Stock_Open, Stock_high, Stock_low, Stock_Close, Stock_volume, Stock_Results) VALUES (?,?,?,?,?,?,?,?)".format(Stock)
        c.execute(values,elements)
        con.commit()

    # If table exists inserting entries to table 
    else:
        values = "INSERT INTO {}(Stock_Date,Stock_Time,Stock_Open, Stock_high, Stock_low, Stock_Close, Stock_volume, Stock_Results) VALUES (?,?,?,?,?,?,?,?)".format(Stock)
        c.execute(values,elements)
        con.commit()

