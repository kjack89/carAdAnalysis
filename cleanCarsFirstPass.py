# -*- coding: utf-8 -*-
"""
@author: kcjac
"""

import re
import time
import math
import pandas as pd
import dateutil
from selenium import webdriver

#choose save location
csvSavePath = 'path_to_csv_file'
#choose document name ending in .csv
docName = 'carsTrucks.csv'
fuelEconomydoc = 'fuelEconomy.csv'

#returns a dictionary of all makes/models of cars from 1984-2020 scraped off of fueleconomy.gov
def makeModelDict():    
    driver = webdriver.Chrome(executable_path=r'C:/chromedriver_win32/chromedriver.exe')
    makeModelDict = {}
    feURL = 'https://www.fueleconomy.gov/feg/findacar.shtml'
    driver.get(feURL)
    time.sleep(5)
    driver.find_element_by_xpath('//*[@id="mnuYear1"]/option[38]').click()
    time.sleep(.5)
    driver.find_element_by_xpath('//*[@id="mnuYear2"]/option[2]').click()
    time.sleep(.5)
    makeList = driver.find_element_by_xpath('//*[@id="mnuMake"]').text.split('\n')
    makeList = makeList[1:len(makeList)]
    makeOption = 1
    for make in makeList:
        makeOption += 1
        driver.find_element_by_xpath('//*[@id="mnuMake"]/option[{}]'.format(makeOption)).click()
        time.sleep(.5)
        makeModelDict.update({make:[]})
        modelList = driver.find_element_by_xpath('//*[@id="mnuModel"]').text.split('\n')
        modelList = modelList[1:len(modelList)]
        for model in modelList:
            makeModelDict[make].append(model)
    driver.quit()
    return makeModelDict

#
def splitDateTime(string):
    return (string[len(string)-5:len(string)])
    
def trimPrice(string):
    string = str(string)
    return re.sub('[\D]','',string)
            
def trimDate(string):
    return (string[:len(string)-5])

def findYear(index):
    index = int(index)
    titleString = str(cars.loc[index]['title'])
    bodyString = str(cars.loc[index]['body'])
    years = [i for i in range(2021, 1950,-1)]
    for year in years:
        year = str(year)
        if re.search(year,titleString) is not None:
            return year
    trimmedYears1 = [i for i in range(0,21)]
    trimmedYears2 = [i for i in range(70,100)]
    trimmedYears = trimmedYears1 + trimmedYears2
    trimmedYears = ['0'+str(i) if len(str(i))==1 else str(i) for i in trimmedYears]
    for year in trimmedYears:
        if re.search(year,titleString) is not None:
            if int(year) <= 20:
                return ('20'+year)
            else:
                return ('19'+year)
    for year in years:
        year = str(year)
        if re.search(year,bodyString) is not None and len(bodyString) < 1000:
            return year
    for year in trimmedYears:
        if re.search(year,bodyString) is not None and len(bodyString) < 1000:
            if int(year) <= 20:
                return ('20'+year)
            else:
                return ('19'+year)
    return 0

def findMake(index, mmDict):
    index = int(index)
    mxlen = 500
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    makes = list(mmDict.keys())
    for val in mmDict.values():
        if '2' in val: val.remove('2')
        if '3' in val: val.remove('3')
        if '5' in val: val.remove('5')
        if '6' in val: val.remove('6')
        if '200' in val: val.remove('200')
        if 'M' in val: val.remove('M')
    if re.search('f-150',titleString) or re.search('f 150',titleString) or re.search('f150',titleString):
        return ('Ford')
    if re.search('chevy',titleString) or re.search('chevy',bodyString):
        return ('Chevrolet')        
    for make in makes:
        if re.search(make.lower(),titleString) is not None:
            return make
        elif re.search((make.lower()+' '),bodyString) is not None and len(bodyString) < mxlen:
            return make
    if re.search('ram',titleString) or re.search('ram',bodyString):
        return('Dodge')
    elif re.search('mercedes',titleString) or re.search('mercedes',bodyString):
        return('Mercedes-Benz')
    elif re.search('vw',titleString) or re.search('vw',bodyString):
        return('Volkswagen')
    else:
        for key, val in mmDict.items():
            for v in val:
                if re.search(v.lower(),titleString) is not None:
                    return key
                elif re.search((v.lower()+' '),bodyString) is not None and len(bodyString) < mxlen:
                    return key

def findModel(index,mmDict):
    index = int(index)
    mxlen = 500
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    if re.search('f-150',titleString) or re.search('f 150',titleString) or re.search('f150',titleString):
        return ('F150')
    if re.search('f-250',titleString) or re.search('f 250',titleString) or re.search('f250,',titleString):
        return ('F250')
    if re.search('silverado',titleString):
        return ('Silverado')
    if re.search('mazda2',titleString) or re.search('mazda 2',titleString) or re.search('mazda2',bodyString) or re.search('mazda 2',bodyString):
        return ('2')
    if re.search('mazda3',titleString) or re.search('mazda 3',titleString) or re.search('mazda3',bodyString) or re.search('mazda 3',bodyString):
        return ('3')
    if re.search('mazda5',titleString) or re.search('mazda 5',titleString) or re.search('mazda5',bodyString) or re.search('mazda 5',bodyString):
        return ('5')
    if re.search('mazda6',titleString) or re.search('mazda 6',titleString) or re.search('mazda6',bodyString) or re.search('mazda 6',bodyString):
        return ('6')
    if re.search('chevy',titleString) :
        titleString = re.sub('chevy','chevrolet',titleString)
    elif re.search('chevy',bodyString) and len(bodyString) < mxlen:
        bodyString = re.sub('chevy','chevrolet',bodyString)
    elif re.search('crv',titleString):
        titleString = re.sub('crv','cr-v',titleString)
    elif re.search('crv',bodyString) and len(bodyString) < mxlen:
        bodyString = re.sub('crv','cr-v',bodyString)
    elif re.search('mercedes',titleString):
        titleString = re.sub('mercedes','mercedes-benz',titleString)
    elif re.search('mercedes',bodyString) and len(bodyString) < mxlen:
        bodyString = re.sub('mercedes','mercedes-benz',bodyString)
    for key, val in mmDict.items():
        if re.search(key.lower(),titleString) is not None:
            for v in val:
                if re.search(v.lower(),titleString) is not None:
                    return v
        elif re.search(key.lower(),bodyString) is not None:
            for v in val:
                if re.search((v.lower()+' '),bodyString) is not None and len(bodyString) < mxlen:
                    return v

def fillOdometer(index):
    odometer = cars.loc[index]['odometer']
    bodyString = str(cars.loc[index]['body']).lower()
    miles = [i for i in range(10,300)]
    miles = [str(i) for i in miles]
    if not math.isnan(odometer):
        if int(odometer) > 300:
            return odometer
        else:
            for mile in miles:
                if re.search(mile+'k',bodyString) or re.search(mile+' k',bodyString):
                    if len(bodyString) < 500:
                        return (int(mile)*1000)
    for mile in miles:
        if re.search(mile+'k',bodyString) or re.search(mile+' k',bodyString):
            if len(bodyString) < 500:
                return (int(mile)*1000)

def fillTitleStatus(index):
    mxlen = 500
    title_status = str(cars.loc[index]['title_status'])
    if title_status != 'nan':
        return title_status.strip()
    title_statuses = ['clean', 'salvage', 'rebuilt', 'lien', 'parts only', 'missing']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    for status in title_statuses:
        if re.search(status,titleString):
            return status
        elif re.search(status,bodyString) and len(bodyString) < mxlen:
            return status

def fillSize(index):
    mxlen = 500
    size = str(cars.loc[index]['size'])
    if size != 'nan':
        return size.strip()
    sizes = ['compact', 'full-size', 'mid-size', 'sub-compact', 'small family car', 'supermini']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()    
    for size in sizes:
        if re.search(size,titleString):
            return size
        elif re.search(size,bodyString) and len(bodyString) < mxlen:
            return size
    
def fillCondition(index):
    mxlen = 500
    condition = str(cars.loc[index]['condition'])
    if condition != 'nan':
        return re.sub('\xa0',' ',condition).strip()
    conditions = ['like new condition', 'excellent condition', 'good condition', 'fair condition', 'new condition', 'as new condition', 'salvage']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    for condition in conditions:
        if re.search(condition,titleString):
            return re.sub(' condition','',condition)
        elif re.search(condition,bodyString) and len(bodyString) < mxlen:
            return re.sub(' condition','',condition)

def fillType(index):
    mxlen = 500
    type1 = str(cars.loc[index]['type'])
    if type1 != 'nan':
        return type1.strip()
    types1 = ['suv','mini-van', 'convertible', 'hatchback', 'truck', 'wagon',
              'sedan', 'van', 'pickup', 'coupe',
              'offroad', 'bus', 'saloon', 'estate']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    if re.search('minivan',titleString) or re.search('mini van',titleString):
        return ('mini-van')
    elif re.search('minivan',bodyString) and len(bodyString) < mxlen:
        return ('mini-van')
    elif re.search('mini van',bodyString) and len(bodyString) < mxlen:
        return ('mini-van')
    for type1 in types1:
        if re.search(type1,titleString):
            return type1
        elif re.search(type1,bodyString) and len(bodyString) < mxlen:
            return type1        

def fillDrive(index):
    mxlen = 500
    drive = str(cars.loc[index]['drive'])
    if drive != 'nan':
        return drive.strip()
    drives = ['fwd', '4wd', 'rwd']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    for drive in drives:
        if re.search(drive,titleString):
            return drive
        elif re.search(drive, bodyString) and len(bodyString) < mxlen:
            return drive
        
def fillTransmission(index):
    mxlen = 500
    transmission = str(cars.loc[index]['transmission'])
    if transmission != 'nan':
        return transmission.strip()
    transmissions = ['automatic', 'manual']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    for transmission in transmissions:
        if re.search(transmission, titleString):
            return transmission
        elif re.search(transmission, bodyString) and len(bodyString) < mxlen:
            return transmission

def fillCylinder(index):
    mxlen = 500
    cylinder = str(cars.loc[index]['cylinders'])
    if cylinder != 'nan':
        return cylinder.strip()
    cylinders = ['4', '6', '8', '10', '5', '12', '3']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    cyls = [i+' cyl' for i in cylinders]
    cylinders = [i+' cylinders' for i in cylinders]
    for cylinder in cylinders:
        if re.search(cylinder, titleString):
            return cylinder
        elif re.search(cylinder, bodyString) and len(bodyString) < mxlen:
            return cylinder
    for cyl in cyls:
        if re.search(cyl, titleString):
            return re.sub('cyl','cylinders',cyl)
        elif re.search(cyl,bodyString) and len(bodyString) < mxlen:
            return re.sub('cyl','cylinders',cyl)

def fillFuel(index):
    mxlen = 500
    fuel = str(cars.loc[index]['fuel'])
    if fuel != 'nan':
        return fuel.strip()
    fuels = ['diesel', 'hybrid', 'electric', 'petrol','gas']
    titleString = str(cars.loc[index]['title']).lower()
    bodyString = str(cars.loc[index]['body']).lower()
    for fuel in fuels:
        if re.search(fuel,titleString):
            return fuel
        elif re.search(fuel, bodyString) and len(bodyString) < mxlen:
            return fuel

def checkMake(index, mmDict):
    make = cars.loc[index]['make']
    model = cars.loc[index]['model']
    modelList = mmDict.get(make)
    if model is None:
        return None
    if make is None:
        return None
    if model in modelList:
        return make
    else:
        return None

def checkModel(index):
    make = cars.loc[index]['make']
    model = cars.loc[index]['model']
    if make is None:
        return None
    else:
        return model

a = makeModelDict()
cars = pd.read_csv(csvSavePath+docName)
cars.rename(columns={'Unnamed: 0':'index','0':'price','1':'city','2':'state','3':'altLoc',
                               '4':'date','5':'title','6':'body','7':'condition','8':'delete','9':'cylinders','10':'drive',
                    '11':'fuel','12':'odometer','13':'size','14':'title_status','15':'transmission','16':'type','17':'color'}, inplace=True)
fueleco = pd.read_csv(csvSavePath+fuelEconomydoc)
fueleco.rename(columns={'Unnamed: 0':'index','0':'make','1':'model','2':'year','3':'config','4':'mpg'}, inplace=True)
del cars['delete']
cars = cars.drop_duplicates(subset=['price','state',
                             'title','body','condition','cylinders','drive',
                             'fuel','odometer','size','title_status','transmission','type','color'], keep='first')
cars['time'] = cars['date'].apply(splitDateTime)
cars['date'] = cars['date'].apply(trimDate)
cars['price'] = cars['price'].apply(trimPrice)
cars = cars.astype({'price':'int64'})
cars['date'] = cars['date'].apply(dateutil.parser.parse, dayfirst=True)
cars['year'] = cars['index'].apply(findYear)
cars = cars.astype({'year':'float64'})
cars = cars[cars['year']!=0]
cars['make'] = cars['index'].apply(findMake, mmDict=a)
cars['model'] = cars['index'].apply(findModel, mmDict=a)
cars['odometer'] = cars['index'].apply(fillOdometer)
cars.astype({'odometer':'float64'})
cars['type'] = cars['index'].apply(fillType)
cars['condition'] = cars['index'].apply(fillCondition)
cars['title_status'] = cars['index'].apply(fillTitleStatus)
cars['size'] = cars['index'].apply(fillSize)
cars['drive'] = cars['index'].apply(fillDrive)
cars['transmission'] = cars['index'].apply(fillTransmission)
cars['cylinders'] = cars['index'].apply(fillCylinder)
cars['fuel'] = cars['index'].apply(fillFuel)
cars['make'] = cars['index'].apply(checkMake, mmDict=a)
cars['model'] = cars['index'].apply(checkModel)
cars.to_csv(csvSavePath+'cleanedCars.csv')