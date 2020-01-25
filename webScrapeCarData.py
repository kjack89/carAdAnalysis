# -*- coding: utf-8 -*-
"""
@author: kcjac
"""
import re
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

#choose save location
csvSavePath = 'path_to_csv_file'

#choose document name ending in .csv
docName = 'carsTrucks.csv'
fuelEconomydoc = 'fuelEconomy.csv'

cityStatesURL = 'https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population'
clPage = '.craigslist.org/d/cars-trucks/search/cta'

#parses html tables from url, extracts tables with headers.
class HTMLTableParser:
    
    def parse_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        html_tables = []
        for table in soup.find_all(class_=re.compile("wikitable sortable")):            
            html_tables.append(self.parse_html_table(table))
        return html_tables
    
    def parse_html_table(self,table):
        n_col = len(table.find_all('th'))
        n_row = len(table.find_all('tr')) - 1
        col_names = []
        
        for th in table.find_all('th'):
            col_names.append(re.sub('(\[.+\])','',th.get_text().rstrip()))            
        
        df = pd.DataFrame(columns = col_names, index = range(0,n_row))
        
        row_marker = 0
        for row in table.find_all('tr'):
            col_marker = 0
            if row_marker == 0:
                row_marker += 1
                continue
            else:
                for col in row.find_all('td'):
                    if re.search('km',col.get_text()) is None:
                        df.iat[row_marker-1,col_marker] = re.sub('(\[.+\])','',col.get_text().rstrip())
                        col_marker += 1
                    else:
                        continue
            row_marker += 1
                
        return df
        
#table of top 250 most populated cities in US    
cityStatesTable = HTMLTableParser().parse_url(cityStatesURL)[0]

clDict ={}
key = 0
start_time = time.time()
#scraping of craigslist begins here
for idx,loc in cityStatesTable.iterrows():
    city = re.sub('[ \.]','',loc['City'])
    url = 'https://'+city+clPage
    nextPage = True
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        while(nextPage):
            rows = soup.find('ul',{'class':'rows'})
            for ad in rows.find_all('li'):
                dateTime = ad.find('time',{'class':'result-date'})['datetime']
                title = ad.find('a',{'class':'result-title hdrlnk'}).get_text()
                try:
                    altLoc = re.sub('[\(\)]','',ad.find('span',{'class':'result-hood'}).get_text())
                except:
                    altLoc = None
                try:
                    price = re.sub('[$]','',ad.find('span',{'class':'result-price'}).get_text())
                except:
                    price = None
                adURL = ad.find('a',{'class':'result-title hdrlnk'})['href']
                adResponse = requests.get(adURL)
                adSoup = BeautifulSoup(adResponse.text,'lxml')
                try:
                    body = adSoup.find('section',{'id':'postingbody'})
                    if body is not None:
                        body = re.sub('(QR Code Link to This Post)','',body.get_text()).lstrip()
                    else:
                        body = None
                except:
                    body = adSoup.find('div',{'class':'removed'})
                    if body is not None:
                        body = re.sub('(QR Code Link to This Post)','',body.get_text()).lstrip()
                    else:
                        body = None
                condition = cylinders = drive = fuel = odometer = size = title_status = transmission = car_type = paint_color = None
                for attributes in adSoup.find_all('span'):
                    string = attributes.get_text().strip()
                    if re.search('condition:',string):
                        condition = re.sub('(condition:)','',string)
                    elif re.search('cylinders:',string):
                        cylinders = re.sub('(cylinders:)','',string)
                    elif re.search('drive:',string):
                        drive = re.sub('(drive:)','',string)
                    elif re.search('fuel:',string):
                        fuel = re.sub('(fuel:)','',string)
                    elif re.search('odometer:',string):
                        odometer = re.sub('(odometer:)','',string)
                    elif re.search('size:',string):
                        size = re.sub('(size:)','',string)
                    elif re.search('title status:',string):
                        title_status = re.sub('(title status:)','',string)
                    elif re.search('transmission:',string):
                        transmission = re.sub('(transmission:)','',string)
                    elif re.search('type:',string):
                        car_type = re.sub('(type:)','',string)
                    elif re.search('paint color:',string):
                        paint_color = re.sub('(paint color:)','',string)
                clDict.update({key:(price,loc['City'],loc['State'],altLoc,dateTime,title,body,condition,
                                   condition,cylinders,drive,fuel,odometer,size,title_status,transmission,car_type,paint_color)})
                key += 1
            nextPage = soup.find('a',{'class':'button next'})['href']
            if nextPage is not None:
                nextPageURL = url+nextPage
                response = requests.get(nextPageURL)
                soup = BeautifulSoup(response.text,'lxml')
            else:
                nextPage = False
    except:
        #print('unable to find page for '+city+','+loc['State'])
        continue

elapsed_time = round((time.time() - start_time)/3600,2)    
df = pd.DataFrame.from_dict(clDict,orient='index')
print('Craigslist car data scraping completed in '+str(elapsed_time)+' hours')
df.to_csv(csvSavePath+docName)

driver = webdriver.Chrome(executable_path=r'C:/chromedriver_win32/chromedriver.exe')
feDict = {}
feURL = 'https://www.fueleconomy.gov/feg/findacar.shtml'
start_time = time.time()
driver.get(feURL)
time.sleep(5)
driver.find_element_by_xpath('//*[@id="mnuYear1"]/option[38]').click()
time.sleep(.5)
driver.find_element_by_xpath('//*[@id="mnuYear2"]/option[2]').click()
time.sleep(.5)
makeList = driver.find_element_by_xpath('//*[@id="mnuMake"]').text.split('\n')
makeList = makeList[1:len(makeList)]
makeOption = 1
key = 0

for make in makeList:
    makeOption += 1
    modelOption = 1
    driver.find_element_by_xpath('//*[@id="mnuMake"]/option[{}]'.format(makeOption)).click()
    time.sleep(.5)
    modelList = driver.find_element_by_xpath('//*[@id="mnuModel"]').text.split('\n')
    modelList = modelList[1:len(modelList)]
    for model in modelList:
        pageNo = 1
        time.sleep(.2)
        make2 = re.sub(' ','%20',make)
        model2 = re.sub(' ','%20',model)
        nextURL = 'https://www.fueleconomy.gov/feg/PowerSearch.do?action=noform&path=1&year1=1984&year2=2021&make='+str(make2)+'&baseModel='+str(model2)+'&srchtyp=ymm&pageno={}&sortBy=Comb&tabView=0&rowLimit=200'.format(pageNo)
        try:
            response = requests.get(nextURL)
            soup = BeautifulSoup(response.text,'lxml')
            nextPage = True
            while(nextPage):
                page = soup.find('table',{'class':'cars display responsive stickyHeader'})
                count1 = 1
                for car in page.find_all('tr',{'class':'ymm-row'}):
                    key += 1
                    year = car.find('a').get_text()[0:4]
                    config = car.find('span',{'class':'config'}).get_text()
                    mpg = 0
                    count2 = 1
                    for m in page.find_all('td',{'class':'mpg-comb'}):
                        if count1 == count2:
                            mpg = m.get_text()
                        count2 += 1
                    count1 += 1
                    feDict.update({key:(make,model,year,config,mpg)})
                nextPage = False
        except:
            continue
        pageNo += 1
driver.quit()
elapsed_time = round((time.time() - start_time)/60,2)
print('Fuel economy data scraping completed in '+str(elapsed_time)+' minutes')
fedf = pd.DataFrame.from_dict(feDict,orient='index')
fedf.to_csv(csvSavePath+fuelEconomydoc)