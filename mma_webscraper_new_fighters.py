# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:30:32 2020

@author: Tigor.Sinuraja_nsp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:15:18 2020

@author: Tigor.Sinuraja_nsp
"""
import urllib
import csv
import pyodbc as db
import sys
import pandas
import requests
#import ICantBelieveItsBeautifulSoup
 
from bs4 import BeautifulSoup




#print(page)


#Parse the html in the 'page' variable, and store it in Beautiful Soup format


#print(soup.title)

#print(soup.title.string)

#alltables=soup.find("tr", class_="table_head" )

#print(alltables)

connection = 'DRIVER={ODBC Driver 13 for SQL Server};SERVER=HP-LAPTOP\SQLSERVER2014;Trusted_Connection=yes;DATABASE=mma_magic'






teller = 1


#file = "C:\data\ufc_fighters_ansi.txt"
#ufc_fighters = pandas.read_csv(file, names=names)



con = db.connect(connection)
            
#names = ['ufc_fighter']
ufc_fighters = pandas.read_sql('''SELECT fighter_name AS ufc_fighter FROM  dbo.mma_event_fighters_v''',con)


cur = con.cursor()
qry = 'USE [mma_magic] DELETE FROM  [dbo].[mma_stats] where fighter_name IN  ( SELECT fighter_name FROM dbo.mma_event_fighters_v)'
print( qry )
cur.execute(qry)
qry = 'USE [mma_magic] DELETE FROM  [dbo].[mma_fighters]  where fighter_name IN  ( SELECT fighter_name FROM dbo.mma_event_fighters_v)'
cur.execute(qry)
qry = 'USE [mma_magic] DELETE FROM  [dbo].[mma_stats_generic]  where fighter_name IN  ( SELECT fighter_name FROM dbo.mma_event_fighters_v)'
cur.execute(qry)

cur.commit() #Use this to commit the insert operation
cur.close()
con.close()  

#file="C:/data/ufc_fighters_ansi.txt"

#ufc_fighters = pandas.read_csv(file, names=names)
print( ufc_fighters )
      
def get_mma_stats(url, fighter_name):  
    print(url)
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    #url = 'http://www.sherdog.com/stats/fightfinder?SearchTxt=&weight=6&association='
   
    try:
        
     

 
        #req1 = urllib.request.Request(url, headers={'User-Agent':user_agent,})
        #response1 = urllib.request.urlopen(req1)
        #page1 = response1.read() 
        #soup1 = BeautifulSoup(page1)
        
        page1 = requests.get(url,headers={'User-Agent':user_agent,})
        soup1 = BeautifulSoup(page1.text, 'html.parser')
  
        
        print("status.....",page1.status_code)
        
        all_links = soup1.findAll('a')  
        
        url2=""
        
        #print("found one? " + fighter_name.replace(' ','-'))
        
        for child in all_links:
       
            if child.attrs['href'].find(fighter_name.replace(' ','-')) != -1:
                #print("found one! " + child.text)
                #print( child.text.strip(), '=>', child.attrs['href'])
                url2 = 'http://www.sherdog.com' + child.attrs['href']
        
        #print ("------------------------->" + url2)
        
       #soup1 = null
        
        fighter_weight=""
        fighter_age =""
        fighter_height = ""
        
        
        #print("status.....",page1.status_code)
            #source = urllib.request.urlopen(url2,headers={'User-Agent':user_agent,}).read()
        page2 = requests.get(url2, headers={'User-Agent':user_agent,})
        soup2 = BeautifulSoup(page2.text, 'html.parser')
        #print("---->>>", soup2)
        
        try:
            print("---->>> begin1")
            strong_list = soup2.find_all('strong' )
            #print("---->>> end1", strong_list[0], strong_list[1])
            for child in strong_list:
                #print("----" + child.text)
                
        except:
            e = sys.exc_info()[0]
            print("general error1", e )
            #raise
        
        #print("---->>@@@>")
        
        #raise
    
        try:   
            print("---->> TEST >")
            for child in strong_list:
                #print("----" + child.text)
                if child.text.find('lbs') != -1:
                    #print(child.string)
                    fighter_weight = child.text
                if child.text.find('AGE') != -1:
                    #print(str(child))
                    fighter_age = child.text
                if child.text.find('"') != -1:
                    #print(str(child))
                    fighter_height = child.text
            print("weight............", fighter_weight)
                    
            con = db.connect(connection)
 
            cur_fighter = con.cursor()
            qry_fighter = '''USE [mma_magic] INSERT INTO [dbo].[mma_fighters]
                           ([fighter_name]
                           ,[fighter_weight]
                           , [fighter_age]
                           )
                     VALUES (? ,?,?)  '''
            try: 
                param_values_fighter = [ str(fighter_name),str(fighter_weight), str(fighter_age) ]
                #print("inside database insert!")
                cur_fighter.execute(qry_fighter, param_values_fighter)
                cur_fighter.commit()
                #print('{0} row inserted fighter successfully.'.format(cur_fighter.rowcount))
            except:
                e = sys.exc_info()[0]
                print("error in insert fighter!", e )
               # raise
                    
             
              
             
                       
            #cur_fighter.commit() #Use this to commit the insert operation
            cur_fighter.close()    
            con.close()  
            
                 
            span=soup2.find_all("span", class_="graph_tag")
            for child in span:
                #print("stats!!!", child.text)
       
                con = db.connect(connection)
                cur = con.cursor()    
                
                qry = '''USE [mma_magic] INSERT INTO [dbo].[mma_stats_generic]
                           ([fighter_name]
                           ,[stats_value]
                           )
                     VALUES (? ,?  )  '''
                param_values = [ fighter_name,str(child )]
                cur.execute(qry, param_values)
                 
                #print('{0} row inserted gneric stats successfully.'.format(cur.rowcount))
                 
                cur.commit() #Use this to commit the insert operation
                cur.close()
                con.close()  
      
            mod_fight_history=soup2.find_all("h2", string ="Fight History - Pro")
            
            #print("module fight history---->" , mod_fight_history[0].parent.parent)
        
          #  div= mod_fight_history.find("table")
            for table in mod_fight_history[0].parent.parent.findChildren("table", recursive=True):
                #print("test----------------------->>")
                # table=child.find("table")
                #print("tr--------------------------------------------------------------------------------", table.text )
                rows = []
                for row in table.find_all("tr"):
                    #for row in rows:
                    #print("td")       
                    rows.append([val.text.encode("utf8") for val in row.find_all("td")])
                    for record in rows:
                        #print("record ----> ", record)
                        dummy = 1
                            
            #with open('C:\data\output_file.csv', 'wb') as f:
                #writer = csv.writer(f)
                     #writer.writerow(headers)
                #writer.writerows(row for row in rows if row)   
 
                #Create connection string to connect DBTest database with windows authentication
                #con = db.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=HP-LAPTOP\SQLSERVER2014;Trusted_Connection=yes;DATABASE=mma_magic')
                    try: 
                        con = db.connect(connection)
                        cur = con.cursor()
                        qry = '''USE [mma_magic] INSERT INTO [dbo].[mma_stats]
                                   ([loss_win]
                                   ,[opponent]
                                   ,[event_name]
                                   ,[way_of_finish]
                                   ,[round_finish]
                                   ,[time_finish])
                             VALUES
                                   (?
                                   ,?
                                   ,? 
                                   ,?
                                   ,?
                                   ,?)
                                '''
                        print("first cursor OK!", cur)
                    except:
                        e = sys.exc_info()[0]
                        print("error in insert 1 stats!", cur)
                        raise
                        
                    param_values = record
                    print(param_values)
                    
                    try: 
                        cur.execute(qry, param_values)
                    except:
                        e = sys.exc_info()[0]
                        print("error in insert stats!", cur)
                        raise
                    cur.commit()
                    cur.close()
                    con.close() 
                    
                    con = db.connect(connection)
                    
                    cur_fighter = con.cursor()
                    qry_fighter = '''USE [mma_magic] UPDATE [dbo].[mma_stats]
                                set[fighter_name] = ?
                               ,[fighter_weight] = ?
                               ,[fighter_age] = ?
                               ,[fighter_height] = ?
                               WHERE fighter_name IS NULL
                              '''
                    try: 
                        param_values_fighter = [ str(fighter_name),str(fighter_weight), str(fighter_age), str(fighter_height) ]
                        print("inside database update")
                        cur_fighter.execute(qry_fighter, param_values_fighter)
                        cur_fighter.commit()
                        print('{0} row updates fighter successfully.'.format(cur_fighter.rowcount))
                    except:
                        e = sys.exc_info()[0]
                        print("error in update fighter!", e)
                        raise
                     
                     
                                 
                    cur_fighter.commit() #Use this to commit the insert operation
                    cur_fighter.close()    
                    con.close()  
                
        
                    #for child1 in child.children:
                    #    print(child1)  
                    #exit
       
        except:
            e = sys.exc_info()[0]
            print("general error", e )
            #raise
   
    except:
        e = sys.exc_info()[0]
        print("general error, waarschijnlijk url", e)
        #raise
 
for index, row in ufc_fighters.iterrows():
    print("fighter: "+row["ufc_fighter"])
    ufc_fighter = row["ufc_fighter"]
    url = 'http://www.sherdog.com/stats/fightfinder?SearchTxt=' + str(ufc_fighter).replace(' ','+') +"&weight=&association="
    fighter_name =  str(ufc_fighter)
    get_mma_stats(url, fighter_name)
    
"""
 
url = 'http://www.sherdog.com/stats/fightfinder?SearchTxt=Aalon+Cruz&weight=&association='
get_mma_stats(url, 'Aalon Cruz')

"""

           
#============================================== K-NEAREST NAIBOUR ================================================================================
  
