import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

import os 


def calculate_mean(df,column_name):
    return df[column_name].mean()

def replace_mis_val_mean(df,column_name):
    mean_val = calculate_mean(df,column_name)
    df.loc[df[column_name].isna(),column_name] = mean_val
    return df 
    


    
def mis_replace_linear_regression(df,X_col=["minPrice"],Y_col="modalPrice"):
    #Missing value Treatment 
    #Method 2 : using function approximation. 
    # Y_col = "modalPrice"
    # X_col = ["minPrice"]
    ################################################################
    df_yes_na = df[df[Y_col].isna()]
    if(len(df_yes_na) == 0):
        print("Nothing to predict, no null values!!!!")
        return df
    
    ################################################################
    # One without no null values in modalprice 
    #Filtering data 
    df_not_na = df[df[Y_col].notna()]
    for x_col_name in X_col:
        df_not_na = df_not_na[df_not_na[x_col_name].notna()]
    # Defining a linear model for mapping x = ["minPrice"], y = "modalPrice"
    X = np.array(np.matrix(df_not_na[X_col]))
    Y = np.array(np.matrix(df_not_na[Y_col]).T) 
    #print(X)
    try: 
        lin_reg   = LinearRegression()
        lin_model = lin_reg.fit(X,Y)
    except:
        print("The Linear regression model didn't run!!!!. Problem with the data. ")
    ################################################################
    ##Replacing the predicted values!!!!....
    X_test    = np.array(np.matrix(df_yes_na[X_col]))
    Y_test    = lin_model.predict(X_test)
    list_index_mis_values = list(df_yes_na.index)
    df.loc[list_index_mis_values,Y_col] = Y_test
    ################################################################
    
    ##testing if null values are replaces using linear regression method. 
    try:
        assert len(df[df[Y_col].isna()])== 0
    except:
        print("The null values are not replaced correctly!!!. Don't execute next code.")
    
    ################################################################
    
    return df





def handle_outlier(df,col_name,val_replace):
    # Outlier Handling. 
    # Going to detect outlier based on standard deviation. if the data point is mean + (3* Std) --> Then classify as outlier
    # treat the point as mean
    #col_name = "mis_m2_minprice"
    mean = df[col_name].mean()
    std  = df[col_name].std()
    out_pos = mean + (3 * std)
    out_neg = mean - (3 * std)
    cond = (df[col_name] > out_pos) | (df[col_name] < out_neg)
    
    #val_replace = mean
    if(len(df[cond])==0):
        print("No outliers to Handle!!!!!!!")
        return df
    else:
        df.loc[cond,col_name] = val_replace
        return df 
    
    


def plot_time_series_data(df,filepath,time_col="timestamp",data_col="min_price",title="Plot showing Min Price Vs Timestamp",plot_show=True,hard_save=False):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'bold',
            'size': 30,
            }


    plt.plot(df[time_col],df[data_col])
    plt.xlabel(time_col,fontdict = font )
    plt.ylabel(data_col,fontdict=font)
    plt.title(title,fontdict=font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    

        
    if(os.path.exists(filepath) & (not hard_save)):
        print("Figure already saved!!!!")
    else:
        print("Saving Figure!!!!")
        plt.savefig(filepath)
        
    
    if plot_show:
        plt.show()

        
def plotting_multiple_series(df,filepath,data_col_list,time_col="timestamp",title="Plot showing Min Price Vs Timestamp",plot_show=True,hard_save=False):

    fig = plt.gcf()
    fig.set_size_inches(16.5, 6.5)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'bold',
            'size': 30,
            }


    y_data_col = "Moving Averages"

    for data_col in data_col_list:
        #print(data_col)
        plt.plot(df[time_col],df[data_col])


    plt.xlabel(time_col,fontdict = font )
    plt.ylabel(y_data_col,fontdict=font)
    plt.title(title,fontdict=font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(data_col_list, loc='upper left')


    if(os.path.exists(filepath) & (not hard_save)):
        print("Figure already saved!!!!")
    else:
        print("Saving Figure!!!!")
        plt.savefig(filepath)


    if plot_show:
        plt.show()



def find_moving_averages(df,col_name,col_name_prefix,window_size_list,generate_col_fun):
    data_col_list = []
    for window_size in window_size_list:
        new_col_name = generate_col_fun(col_name_prefix,window_size)
        data_col_list+=[new_col_name]
        df[new_col_name]= df[col_name].rolling(window=window_size).mean()
    
    return df,data_col_list





def plot_forecasted_validation(data_dict):
    # Need to work on this
    ####
    df_train = data_dict["df_train"] 
    df_test  = data_dict["df_test"]
    a_col_name_train = data_dict["a_col_name_train"]
    a_col_name_test  = data_dict["a_col_name_test"]
    f_col_name       = data_dict["f_col_name"]
    time_col         = data_dict["time_col"]
    y_data_col       = data_dict["y_data_col"]
    title            = data_dict["title"]
    file_path        = data_dict["file_path"]
    hard_save        = data_dict["hard_save"]
    plot_show        = data_dict["plot_show"]
    #######
    
    fig = plt.gcf()
    fig.set_size_inches(16.5, 6.5)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'bold',
            'size': 30,
            }
    plt.plot(df_train[a_col_name_train],label="Train")
    plt.plot(df_test[a_col_name_test],label ="Test")
    plt.plot(df_test[f_col_name],label="Forecasted")
    
    
    
    plt.xlabel(time_col,fontdict = font )
    plt.ylabel(y_data_col,fontdict=font)
    plt.title(title,fontdict=font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    
    if(os.path.exists(file_path) & (not hard_save)):
        print("Figure already saved!!!!")
    else:
        print("Saving Figure!!!!")
        plt.savefig(file_path)


    if plot_show:
        plt.show()
    
    












## HELPER FUNCTIONS
def convert_list_to_str(x):
    res = list(map(str, x))
    new_window='_'.join(res)
    return new_window







