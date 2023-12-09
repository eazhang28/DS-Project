import pandas
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('miniproject-f23-eazhang28/nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data

# Numerical Conversions
day_to_numeric = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
months_to_numeric = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Turn Weekdays into Numeric Values
dataset_2['Weekday'] = dataset_2['Day'].map(day_to_numeric)

# Turn Date into Months without the Days
dataset_2[['Day', 'Month']] = dataset_2['Date'].str.split('-', expand=True)
dataset_2.drop('Date', axis=1, inplace=True)
dataset_2.drop('Day',axis=1, inplace=True )
dataset_2['Month'] = dataset_2['Month'].map(months_to_numeric)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Which Three Bridges Should We Install Sensors On?
# Features: Brooklyn Bridge, Manhattan Bridge, Queesboro Bridge, Williamsburg Bridge
# Target: Total

mse1 = []
coefdet1 = [] 

X1 = dataset_2[['Brooklyn Bridge','Manhattan Bridge','Queensboro Bridge']].to_numpy()

intercepts = np.ones((X1.shape[0],1))
X1 = np.hstack((X1,intercepts))

y1 = dataset_2[['Total']].astype(int).to_numpy()

X1_train, X1_test, y1_train, y1_test, = tts(X1,y1,test_size=.2,random_state=61)

reg1 = Ridge(alpha=0.5)
reg1.fit(X1_train,y1_train)
y1_pred = reg1.predict(X1_test)
mse1.append(mean_squared_error(y1_test, y1_pred))
coefdet1.append(r2_score(y1_test, y1_pred))
plt.figure(1)
plt.plot(y1_test,'*')
plt.plot(y1_pred)
plt.title('Brooklyn, Manhattan, and Queensboro Bridges Features')
plt.xlabel('Time (Days)')
plt.ylabel('Traffic (Count of Bicyclists)')
plt.show()

X1 = dataset_2[['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge']].to_numpy()

intercepts = np.ones((X1.shape[0],1))
X1 = np.hstack((X1,intercepts))

y1 = dataset_2[['Total']].astype(int).to_numpy()

X1_train, X1_test, y1_train, y1_test, = tts(X1,y1,test_size=.2,random_state=61)

reg1 = Ridge(alpha=0.5)
reg1.fit(X1_train,y1_train)
y1_pred = reg1.predict(X1_test)
mse1.append(mean_squared_error(y1_test, y1_pred))
coefdet1.append(r2_score(y1_test, y1_pred))
plt.figure(2)
plt.plot(y1_test,'*')
plt.plot(y1_pred)
plt.title('Brooklyn, Manhattan, and Williamsburg Bridges Features')
plt.xlabel('Time (Days)')
plt.ylabel('Traffic (Count of Bicyclists)')
plt.show()

X1 = dataset_2[['Brooklyn Bridge','Queensboro Bridge','Williamsburg Bridge']].to_numpy()

intercepts = np.ones((X1.shape[0],1))
X1 = np.hstack((X1,intercepts))

y1 = dataset_2[['Total']].astype(int).to_numpy()

X1_train, X1_test, y1_train, y1_test, = tts(X1,y1,test_size=.2,random_state=61)

reg1 = Ridge(alpha=0.5)
reg1.fit(X1_train,y1_train)
y1_pred = reg1.predict(X1_test)
mse1.append(mean_squared_error(y1_test, y1_pred))
coefdet1.append(r2_score(y1_test, y1_pred))
plt.figure(3)
plt.plot(y1_test,'*')
plt.plot(y1_pred)
plt.title('Brooklyn ,Queensboro, and Williamsburg Bridges Features')
plt.xlabel('Time (Days)')
plt.ylabel('Traffic (Count of Bicyclists)')
plt.show()

X1 = dataset_2[['Manhattan Bridge','Queensboro Bridge','Williamsburg Bridge']].to_numpy()

intercepts = np.ones((X1.shape[0],1))
X1 = np.hstack((X1,intercepts))

y1 = dataset_2[['Total']].astype(int).to_numpy()

X1_train, X1_test, y1_train, y1_test, = tts(X1,y1,test_size=.2,random_state=61)

reg1 = Ridge(alpha=0.5)
reg1.fit(X1_train,y1_train)
y1_pred = reg1.predict(X1_test)
mse1.append(mean_squared_error(y1_test, y1_pred))
coefdet1.append(r2_score(y1_test, y1_pred))
plt.figure(4)
plt.plot(y1_test,'*')
plt.plot(y1_pred)
plt.title('Manhattan, Queensboro, and Williamsburg Bridges Features')
plt.xlabel('Time (Days)')
plt.ylabel('Traffic (Count of Bicyclists)')
plt.show()


print(mse1)
print(coefdet1)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# What Weather Conditions Should We Focus On?
# Features: High Temp, Low Temp, Precipitation
# Target: Total

X2 = dataset_2[['High Temp', 'Low Temp', 'Precipitation']].to_numpy()

intercepts = np.ones((X2.shape[0],1))
X2 = np.hstack((X2,intercepts))

y2 = dataset_2[['Total']].astype(int).to_numpy()

X2_train, X2_test, y2_train, y2_test, = tts(X2,y2,test_size=.3,random_state=11)

reg2 = Ridge(alpha=0.75)
reg2.fit(X2_train,y2_train)
y2_pred = reg2.predict(X2_test)
mse2 = mean_squared_error(y2_test, y2_pred)
coefdet2 = r2_score(y2_test, y2_pred)
print('Q2 MSE: %f' % mse2)
print('Q2 R2: %f' % coefdet2)
plt.figure(5)
plt.plot(y2_test,'*')
plt.plot(y2_pred)
plt.title('Weather Conditions')
plt.xlabel('Time (Days)')
plt.ylabel('Traffic (Count of Bicyclists)')
plt.show()



# What Day is it Based on the Traffic Numbers?
# Features: Total
# Target: Weekday

X3 = dataset_2[['Total']].to_numpy()

intercepts = np.ones((X3.shape[0],1))
X3 = np.hstack((X3,intercepts))

y3 = dataset_2[['Weekday']].astype(int).to_numpy()

X3_train, X3_test, y3_train, y3_test, = tts(X3,y3,test_size=.2,random_state=42)

reg3 = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)
reg3.fit(X3_train,y3_train)
y3_pred = reg3.predict(X3_test)
accuracy = accuracy_score(y3_test, y3_pred)
print('Q3 Accuracy: %f' % accuracy)
plt.figure(6)
plt.plot(y3_test,'*')
plt.plot(y3_pred)
plt.xlabel('Traffic (Count of Bicyclists)')
plt.ylabel('Day of The Week')
plt.show()

# print(dataset_2.to_string())