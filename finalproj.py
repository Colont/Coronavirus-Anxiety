import os 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor

os.chdir(r'\Users\cjbea\Documents\ld\New folder')
#print(os.getcwd())

df_loneliness = pd.read_excel(
    'coronavirusandanxietyestimates3apr10may2020.xlsx',
                              sheet_name='Loneliness', skiprows=4)
df_gender = pd.read_excel(
    'coronavirusandanxietyestimates3apr10may2020.xlsx',
                              sheet_name='Sex', skiprows=4)
df_work_affected = pd.read_excel(
    'coronavirusandanxietyestimates3apr10may2020.xlsx',
                              sheet_name='Work affected', skiprows=4)
df_deaths = pd.read_excel('datadownload.xlsx', sheet_name='deaths', skiprows=6)
def rows_lonely(df):
    
    dict1 = {}
    
    for x in range(1, 8):  # Assuming you want to iterate from the second row (index 1) to the eighth row (index 7)
        df_row = df_loneliness.loc[x].to_dict()
        mean = df_row['Mean average']
        mean1 = df_row['Mean average.1']
        mean2 = df_row['Mean average.2'] 
        mean3 = df_row['Mean average.3']
        mean4 = df_row['Mean average.4']
        dict1[x] = {
            'Mean average': mean,
            'Mean average.1': mean1,
            'Mean average.2': mean2,
            'Mean average.3': mean3,
            'Mean average.4': mean4
            }
    means = {}
    for key, value_dict in dict1.items():
    
        values = list(value_dict.values())
        mean_val = sum(values)/ len(values)
        means[key] = mean_val
    
    x_val = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 
             'Week 6', 'Week 7']
    y_val = list(means.values())
   
    data = {'Week': range(1,8),
            'Means': y_val}
    df = pd.DataFrame(data)
    #plot the average loneliness values over a period of time
    #of 7 weeks
    plt.plot(x_val, y_val, marker='o', linestyle='-', color='b', 
             label='Mean Values')
    plt.ylabel('Anxiety values out of 10')
    plt.title('Mean Values Over Time')
    plt.legend()
    plt.show()
    #make a dicitonary to turn into dataframe for the lonliness values
    data = {'Week': range(1,8),
            'Means': y_val}
    df = pd.DataFrame(data)
    '''
    using pearsonr find rho and p value
    '''
    
    cor, p_value = stats.pearsonr(df['Week'], np.log(df['Means']))


 
    X = df[['Week']]
    y = df['Means']
   
    
   #ensure no negative predictions use poisson regressor
    model = PoissonRegressor()
    model.fit(X,y)
    y_pred =model.predict(X)
    #plot best fit model for the data
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred, color='red', label='Fitted Line')
    plt.xlabel('Weeks')
    plt.ylabel('Anxiety values')
    plt.title('Anxiety best fit over weeks in COVID')
    plt.legend()
    plt.show()
    #linear regression plot
    r = np.corrcoef(df['Week'], y)
    #r values of the data
    
   
    #values
    
    #predicting the values today
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.15, 
                                                        random_state = 1)
    
    #explain why this is not a good representation however.
    
    model.fit(X_train,y_train)
    y_predicted = model.predict([[8],[9],[10],[11],[12],[13],[14]])
    dict_pred = {"Predicted": y_predicted}
    df_pred = pd.DataFrame(dict_pred)
   
    return df_pred, r, cor, p_value
'''
Sort by column
'''

def col_lonely(df):
    
  #set values for columns due to the columns being formatted weirdly on
  #excel
    df = {'Never lonely': df['Mean average'],
          'Hardly ever lonely': df['Mean average.1'],
          'Occasionally lonely': df['Mean average.2'],
          'Some of the time lonely': df['Mean average.3'],
          'Often/Always Lonely': df['Mean average.4']}
    
    df = pd.DataFrame(df)
    df= df.head(8)
    df = df.dropna()
    #sorting the data to get the rows only
    df['Week'] = range(1,8)
    X = df[['Week']]
    y_never = df['Never lonely']
    y_hardly = df['Hardly ever lonely']
    y_occasionally = df['Occasionally lonely']
    y_some_of_time = df['Some of the time lonely']
    y_always = df['Often/Always Lonely']
    #plot each never and always
    plt.plot(X, y_never, marker='o', label='Never')
    plt.plot(X, y_always, marker='o', label='Always')
    plt.xlabel('Weeks')
    plt.ylabel('Mean Anxiety of out 10')
    plt.title('Anxiety for loneliness over weeks covid')
    plt.legend()
    plt.show()
    #we indentified for the mean averages there is a correlation
    #between all the groups, but how about individual groups?
  
    model = PoissonRegressor()
    model.fit(X, y_always)
    y_pred_always =model.predict(X)
    #best fit for always
    plt.scatter(X, y_always, color='blue', label='Data Points')
    plt.plot(X, y_pred_always, color='red', label='Fitted Line')
    plt.xlabel('Weeks')
    plt.ylabel('Anxiety out of 10')
    plt.title('Anxiety for loneliness always over weeks in COVID best fit')
    plt.legend()
    plt.show()
    
    r_always = np.corrcoef(df['Week'], y_always)
    

    
    #find r val for r_never
    #best fit for never
    model.fit(X, y_never)
    y_pred_never = model.predict(X)
    plt.scatter(X, y_never, color='blue', label='Data Points')
    plt.plot(X, y_pred_never, color='red', label='Fitted Line')   
    plt.xlabel('Weeks')
    plt.ylabel('Anxiety out of 10')
    plt.title('Anxiety for loneliness never over weeks in COVID best fit')
    plt.legend()
    plt.show()
    
    sns.residplot(x=[1,2,3,4,5,6,7], y=y_never)
    plt.show()
    r_never = np.corrcoef(df['Week'], y_never)
    
    #after r value comparison it appears that these two
    #groups have a vast difference in the correlation
    return r_never, r_always
def anxiety_sex(df):
    df = {'Male mean': df['Mean average'],
          'Female mean': df['Mean average.1']}
    weeks = range(1,8)
    df = pd.DataFrame(df)
    df = df.dropna()
    #plot anxiety by gender after sorting the data
    plt.plot(weeks, df['Male mean'], label = 'Male mean', marker='o')
    plt.plot(weeks, df['Female mean'], label = 'Female mean', marker='o')
    plt.xlabel('Weeks')
    plt.ylabel('Anxiety out of 10')
    plt.title('Anxiety male and female over weeks in COVID')
    plt.legend()
    plt.show()
   
   
    
  
def work_aff(df):
    df = {'Work not affected': df['Mean average'],
          'Work affected': df['Mean average.1']}
    df=pd.DataFrame(df)
    df=df.head(8)
    df=df.dropna()
    df.replace('x', pd.NA, inplace=True)
    df=df.dropna()
   
    weeks = range(2,8)
    plt.plot(weeks, df['Work not affected'], label='Work not affected',
             marker='o')
    plt.plot(weeks, df['Work affected'], label='Work affected',
             marker='o')
    plt.xlabel('Weeks')
    plt.ylabel('Anxiety out of 10')
    plt.title('Anxiety for work impact')
    plt.legend()
    plt.show()
def deaths(df):
    plt.bar(df['Week no.'],df['COVID-19'])
    plt.xlabel('Weeks')
    plt.ylabel('Daily cases')
    plt.xlim(12,26)
    plt.title('Daily Deaths COVID')

def main():
    df_pred, r_lonely, cor, p_value = rows_lonely(df_loneliness)
    print(f"Predicted for weeks 8 to 15 are \n {df_pred}")
    print(f"Correlation coefficient for all lonely is {cor}")
    print(f"P value for lonely is {p_value}, so we reject the null")
    r_never, r_always = col_lonely(df_loneliness)
    print(f"Correlation coefficient for never lonely is {r_never[0,1]}")
    print(f"Correlation coefficient for always lonely is {r_always[0,1]}")
    anxiety_sex(df_gender)
    work_aff(df_work_affected)
    deaths(df_deaths)
if __name__ == '__main__':
    main()
