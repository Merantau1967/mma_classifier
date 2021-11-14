# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:52:07 2020

@author: Tigor.Sinuraja_nsp
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 19:58:25 2018

@author: Tigor
"""

""

#import the library used to query a website
import urllib
import csv
import pyodbc as db
import sys
import pandas
import requests

from bs4 import BeautifulSoup


connection = 'DRIVER={ODBC Driver 13 for SQL Server};SERVER=HP-LAPTOP\SQLSERVER2014;Trusted_Connection=yes;DATABASE=mma_magic'


#print(page)


#Parse the html in the 'page' variable, and store it in Beautiful Soup format


#print(soup.title)

#print(soup.title.string)

#alltables=soup.find("tr", class_="table_head" )

#print(alltables)

con = db.connect(connection)

cur = con.cursor()
qry = '''USE [mma_magic] DELETE FROM  [dbo].[mma_event_fights]'''
cur.execute(qry)

cur.commit() #Use this to commit the insert operation
cur.close()
con.close()  



teller = 1


#file = "C:\data\ufc_fighters_ansi.txt"
#ufc_fighters = pandas.read_csv(file, names=names)


con = db.connect(connection)
            
#names = ['ufc_fighter']
#ufc_fighters = pandas.read_sql('''SELECT fighter_name AS ufc_fighter FROM dbo.ufc_fighters_v''',con)

#file="C:/data/ufc_fighters_ansi.txt"

#ufc_fighters = pandas.read_csv(file, names=names)
#print( ufc_fighters )
      
def get_mma_events(url):  
    print(url)
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    #url = 'http://www.sherdog.com/stats/fightfinder?SearchTxt=&weight=6&association='
    page = requests.get(url,headers={'User-Agent':user_agent,})
    soup = BeautifulSoup(page.text, 'html.parser')
        
    print(" step 1 ")
        
    print("status.....", page.status_code)
    #print("text.....", page.text)
    
    try:
        
     
      #  print("found one? ",  all_pairs)
      
        fighter_1 =''
        fighter_2 =''
        
        module_event_match = soup.find_all("div", class_ ="module event_match")
        
        #print("---->", module_event_match)
        
        for row in   module_event_match[0].findChildren("tr", recursive=True):
            
            #print(" ",row)
            dummy = 0
            #print("record ----> ", row.text)
            for val in row.findChildren("meta",itemprop="name"):

                dummy +=1
                print("dummy", dummy)
                print( "val --- > ", val.get("content"))
                event = val.get("content")
                if (dummy == 2):
                    fighter_1 = val.text
                    print( "fighter_1", val.text)
                elif (dummy == 4):
                    fighter_2 = val.text
                    print("fighter_2", val.text)
                try: 
                      
                                    
                    con = db.connect(connection)
                 
                    cur_fighter = con.cursor()
                    qry_fighter = '''USE [mma_magic] INSERT INTO [dbo].[mma_event_fights]
                                           ([fighter_1]
                                           ,[fighter_2]
                                           , [event]
                                           )
                                     VALUES (? ,?,?)  '''
                    try: 
                        param_values_fighter = [fighter_1, fighter_2,event ]
                                #print("inside database insert!")
                        cur_fighter.execute(qry_fighter, param_values_fighter)
                        cur_fighter.commit()
                                #print('{0} row inserted fighter successfully.'.format(cur_fighter.rowcount))
                    except:
                        e = sys.exc_info()[0]
                        print("error in insert fighter!", qry_fighter )
                        raise
                            
                     
                      
                     
                               
                    #cur_fighter.commit() #Use this to commit the insert operation
                    cur_fighter.close()    
                    con.close()  
                    
                         
                
               
                except:
                    e = sys.exc_info()[0]
                    print("general error" )
                         #   raise
   
    except:
       e = sys.exc_info()[0]
       print("general error, waarcshijnlijk url")
        #raise
 

#for index, row in ufc_fighters.iterrows():
#    print("fighter: "+row["ufc_fighter"])
#    ufc_fighter = row["ufc_fighter"]
    
 #   https://www.sherdog.com/events/UFC-Fight-Night-168-Felder-vs-Hooker-82983
    
url = 'https://www.sherdog.com/events/UFC-Fight-Night-169-Benavidez-vs-Figueiredo-83479'

get_mma_events(url)
 
#url = 'http://www.sherdog.com/stats/fightfinder?SearchTxt=Marc+Diakiese&weight=&association='
#get_mma_stats(url, 'Marc Diakiese')

           
#============================================== K-NEAREST NAIBOUR ================================================================================
  