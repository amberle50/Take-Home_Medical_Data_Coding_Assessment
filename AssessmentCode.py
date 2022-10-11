# -*- coding: utf-8 -*-
"""
Coding Exercise
Created on Thu Sep 15 08:57:05 2022

@author: Amberle McKee
"""

#Import the necessary libraries and modules
import pandas as pd #This library allows me to work with dataframes (tables) easily.
import numpy as np #This library allows me to easily run calculations.
import datetime#This library allows me to work easily with dates
from datetime import date#This module has important date formatting rules
import icd #This library allows me to work with ICD codes and comorbidities easily.
#NOTE: To import icd, you may need to install the package by typing the following 
#into the IPython console: pip install icd
from sklearn.linear_model import LogisticRegression#This module will allow me to easily perform a logistic regression
from sklearn import metrics# import the metrics class to evaluate the model
import matplotlib.pyplot as plt #This library allows me to easily plot data

#Import the csv file
data=pd.read_csv('G:/My Drive/Job Applications/Software Engineering Industry/Take-Home Assessments/USNaWR_CodingAssessment/Take-Home_Medical_Data_Assessment/USN_claims_test_data.csv')

#Get some basic info on the layout of the dataset
print(data.head)
print(data.info())
print(data.describe())

## TASK 1: Subset the dataset to the population at risk, admissions for patients undergoing isolated
##coronary artery bypass grafts (CABG), identified by procedure codes of 3160, 3611, 3612,
##3613, 3614, 3615, 3616, 3617, or 3619.

#First I will create 5 arrays that contain only the data in the rows that contain
#at least one of those procedure codes.
p1=data[data['procedure1'].isin([3160,3611,3612,3613,3614,3615,3616,3617,3619])]
p2=data[data['procedure2'].isin([3160,3611,3612,3613,3614,3615,3616,3617,3619])]
p3=data[data['procedure3'].isin([3160,3611,3612,3613,3614,3615,3616,3617,3619])]
p4=data[data['procedure4'].isin([3160,3611,3612,3613,3614,3615,3616,3617,3619])]
p5=data[data['procedure5'].isin([3160,3611,3612,3613,3614,3615,3616,3617,3619])]

#Then I will concatenate these into a new dataframe.
risky_pop=pd.concat([p1,p2,p3,p4,p5])#creates a new dataframe of only the at risk population

#Creates a new column in data specifying whether the patient is in the at risk population
data['is_risky?']=np.nan
for i in data.index:
    if i in risky_pop.index:
        data['is_risky?'][i]=1
    else:
        data['is_risky?'][i]=0
        
print('For Task 1, patients admitted for CABG surgery are stored in a dataset called risky_pop. There is also a column in the data dataframe labeled is_risky? that denotes whether a patient in that dataframe is admitted for CABG surgery.')

##TASK 2: Identify comorbidities present in the diagnosis codes using either the Elixhauser (available
##from AHRQ for SAS, or on SSC for Stata) or Charlson index (either is fine for this exercise).

#I'm going to use a method which requires the ICD codes to be strings instead of integers.
#So I am going to create a new dataframe in which the codes are strings so that I keep the 
#original dataframe in integers for future analyses. This new dataframe will be called data2.
data2=data.copy()#copy dataframe to data2
data2.diagnosis1 = [str(i) for i in data2.diagnosis1]#make diagnosis1 column into strings
data2.diagnosis2 = [str(i) for i in data2.diagnosis2]#make diagnosis2 column into strings
data2.diagnosis3 = [str(i) for i in data2.diagnosis3]#make diagnosis3 column into strings
data2.diagnosis4 = [str(i) for i in data2.diagnosis4]#make diagnosis4 column into strings
data2.diagnosis5 = [str(i) for i in data2.diagnosis5]#make diagnosis5 column into strings


#Specify the ICD-9 codes for the charlson index comorbidity mapping
#The defaults are in ICD-10, so I need to make a custom map for ICD-9 codes.
charlson9 = {
"myocardial_infarction":['410','412'],
"congestive_heart_failure":['39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4254', '4255', '4257', '4258', '4259', '428'],
"periphral_vascular_disease":['0930', '4373', '440', '441', '4431', '4432', '4438', '4439', '4471', '5571', '5579', 'V434'],
"cerebrovascular_disease":['36234', '430', '431', '432', '433', '434', '435', '436', '437', '438'],
"dementia":['290', '2941', '3312'],
"chronic_pulmonary_disease":['4168', '4169', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505', '5064', '5081', '5088'],
"connective_tissue_disease_rheumatic_disease":['4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '7148', '725'],
"peptic_ulcer_disease":['531', '532', '533', '534'],
"mild_liver_disease":['07022', '07023', '07032', '07033', '07044', '07054', '0706', '0709', '570', '571', '5733', '5734', '5738', '5739', 'V427'],
"diabetes_wo_complications":['2500', '2501', '2502', '2503', '2508', '2509'],
"diabetes_w_complications":['2504', '2505', '2506', '2507'],
"paraplegia_and_hemiplegia":['3341', '342', '343', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449'],
"renal_disease":['40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493', '582', '5830', '5831', '5832', '5834', '5836', '5837', '585', '586', '5880', 'V420', 'V451', 'V56'],
"cancer":['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '200', '201', '202', '203', '204', '205', '206', '207', '208', '2386'],
"moderate_or_sever_liver_disease":['4560', '4561', '4562', '5722', '5723', '5724', '5728'],
"metastitic_carcinoma":['196', '197', '198', '199'],
"aids_hiv":['042', '043', '044']
}

#Map the comorbidities to every entry in the dataset
comorbidity_map=icd.icd_to_comorbidities(data2,'patientId',['diagnosis1','diagnosis2','diagnosis3','diagnosis4','diagnosis5'], mapping=charlson9)

#Some patients have multiple entries. So here we'll ensure every patient has only single entries for each of their diagnoses.
#For example, if a patient is diagnosed with diabetes at three visits, this will reduce that down to one diagnoses of diabetes.
unique_diagnoses=icd.long_to_short_transformation(data2,'patientId',['diagnosis1','diagnosis2','diagnosis3','diagnosis4','diagnosis5'])

#The following grabs the column names of the unique_diagnoses dataframe.
num_diagnoses=[str(i) for i in unique_diagnoses.columns]
num_diagnoses=num_diagnoses[0:-1]#Removes the last column from this list.

#This maps the comorbidities to every unique diagnosis, giving us a list of comorbidities for every patient.
unique_comorbidities=icd.icd_to_comorbidities(unique_diagnoses,'patientId',num_diagnoses,mapping=charlson9)

#The following for loop creates a new column in the original data dataframe.
#This column will be the number of comorbidities each patient has.
data['Num_of_Comorbidities']=np.nan#Creates new column and fills it with nans.
for i in data.index:
    tmp=unique_comorbidities[unique_comorbidities['patientId']==data['patientId'][i]]#Find the patient's data
    tmp=tmp.iloc[:,:-1]#Removes the patientId column so it doesn't get added to the subsequent sum
    data['Num_of_Comorbidities'][i]=sum(tmp.sum())#adds the number of comorbidities for that patient to the new column 
    
#Since we'll need this column in our risky_pop dataframe later on, I'm just going to 
#repeat this loop here for the risky_pop dataframe.
risky_pop['Num_of_Comorbidities']=np.nan#Creates new column and fills it with nans.
risky_pop=risky_pop.reset_index()#This resets the index and moves the current index over by 1.
for i in risky_pop.index:
    tmp=unique_comorbidities[unique_comorbidities['patientId']==risky_pop['patientId'][i]]#Find the patient's data
    tmp=tmp.iloc[:,:-1]#Removes the patientId column so it doesn't get added to the subsequent sum
    risky_pop['Num_of_Comorbidities'][i]=sum(tmp.sum())#adds the number of comorbidities for that patient to the new column 
  
print('For Task 2, the comorbidities of each patient are mapped in the dataframe unique_comorbidities. Columns are also added to the data and risky_pop dataframes with the number of comorbidities per patient. Comorbidities were identified with the Charlson Index.')

##TASK 3: Identify whether each admission involved a readmission. A readmission here is defined as a
##subsequent hospitalization for the same patientId within 30 days of the index admission.

#The following changes the format of the admitDate column from dd-mmm-yy to dd, mm, yy
data.admitDate=[datetime.datetime.strptime(i,'%d-%b-%y').strftime('%y, %m, %d') for i in data.admitDate]
   
 
#Function to report the number of days between two dates
def numOfDays(date1, date2):
    return (date2-date1).days #subtract the dates and make the output days
     
#Now I'll add a new column to the dataset with information about whether the patient was readmitted.
data['readmitted?']=np.nan #creates a new column specifying whether this hospital visit resulted in a readmission within 30 days
for row in data.index[1:]:
    if data.patientId[row] == data.patientId[row-1]:#compare current row's patientId to previous row's patientId
        d1=data.admitDate[row-1]#store previous visit date
        d2=data.admitDate[row]#store current visit date
        date1 = date(int(d1[0:2]), int(d1[4:6]), int(d1[8:10]))#transform date into int format
        date2 = date(int(d2[0:2]), int(d2[4:6]), int(d2[8:10]))#transform date into int format
        admission_delay=numOfDays(date1, date2)#calculate the number of days between admissions
        if admission_delay <30:#determine if there are <30 days between admissions
            data['readmitted?'][row-1]=1
        else:
            data['readmitted?'][row-1]=0
    else:
        data['readmitted?'][row-1]=0

#Since we don't have more data after the last entry, I'll assume the last entry was not subsequently readmitted.
data['readmitted?'].iloc[-1]=0 #Marks the last entry as not readmitted.

#Since we'll need this column added to the risky_pop dataframe later, I'll add it now.
risky_pop['readmitted?']=np.nan#add an empty column to the dataframe
for i in risky_pop.index:#for every value in risky_pop
    risky_pop['readmitted?']=data['readmitted?'][risky_pop['index'][i]]#add the corresponding value from the readmitted? column in data

print('For Task 3, a new column entitled readmitted? was added to both the data and risky_pop dataframes with a 1 if the patient was readmitted within 30 days and a 0 if they were not.')

##TASK 4: Specify and run a regression model that estimates the likelihood of readmission among
##patients admitted for CABG surgery. Control for age, systolic blood pressure, and the number of
##comorbidities present in the admission record.

#The age column is missing values. I will impute those as 0, so the regression will run.
data['age']=data['age'].fillna(0)
risky_pop['age']=risky_pop['age'].fillna(0)

#Visually check for collinearity among independent variables
plt.figure()
plt.scatter(data.age,data.systolic)
plt.xlabel('Age')
plt.ylabel('Systolic BP')
plt.show()

plt.figure()
plt.scatter(data.age,data.Num_of_Comorbidities)
plt.xlabel('Age')
plt.ylabel('Number of Comorbidities')
plt.show()

plt.figure()
plt.scatter(data.Num_of_Comorbidities,data.systolic)
plt.xlabel('Number of Comorbidities')
plt.ylabel('Systolic BP')
plt.show()

#Set up my parameters for the logistic regression
 #The x parameter will include the individual, their age, blood pressure, and number of comorbidities
x=np.array([data['patientId'],data['age'],data['systolic'],data['Num_of_Comorbidities']]).reshape(-1,4)
 #The y parameter will be a binary for whether they were readmitted within 30 days.
y = np.array([int(i) for i in data['readmitted?']])

#I will train the model on the full dataset to establish the correct coefficients.
model = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=0).fit(x,y)

#Now I will set up my risky_pop x parameter.
x2=np.array([risky_pop['patientId'],risky_pop['age'],risky_pop['systolic'],risky_pop['Num_of_Comorbidities']]).reshape(-1,4)

#Now I will run the trained logistic regression model on the risky_pop subset (those admitted for CABG surgery)
results=model.predict(x2)

cnf_matrix = metrics.confusion_matrix(risky_pop['patientId'], results)
cnf_matrix

#results[:,1] gives me each patient's likelihood of readmission. 
#Below, I will find the mean of these likelihoods to get an overall likelihood of readmission in this population.
prob_results=model.predict_proba(x2)
prob_readmitted=np.mean(prob_results[:,1])
print('For Task 4, I ran a logistic model and found that the likelihood of a patient admitted for CABG surgery to be readmitted within 30 days was ' + str(prob_readmitted) + '.')