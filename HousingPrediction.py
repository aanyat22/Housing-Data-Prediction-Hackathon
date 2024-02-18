import pandas as pd
import sklearn
#from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



#main method
def main():

    #getting required files for input
    inputted= input('Enter the file for the training data')
    userData= input('Input the file of the house data for the house price(s) you want to predict')

    #reading the files
    input_df = pd.read_csv(inputted)
    user_df1 = pd.read_csv(userData)
    user_df = pd.read_csv(userData)
    user_df.drop(columns=['id','price'], inplace=True)

    #encoding the data 
    encode_dataset(input_df)
    encode_dataset(user_df)

    #printing out predicted values
    print("Here are the predicted prices:")
    predictions= find_regression(input_df,user_df)

    user_df1['price']=predictions
    print(user_df1)
    user_df1.to_csv('TestData.csv',index=False)
    


#encodes the string data of categorical variables into numeric values for feature preprocessing for a given data column
def encode_data(data_column,input_df):
    #find the different non-integer values in the data
    label_values = input_df[data_column].unique()
    df = pd.DataFrame(label_values)
   
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Perform label encoding
    input_df[data_column] = label_encoder.fit_transform(input_df[data_column])

#iterates through a dataset and encodes only categorical variables
def encode_dataset(input_df:pd.DataFrame):
    #gets list of column names
    columnsList = input_df.columns.values
    print(columnsList)
    for i in range(len(columnsList)):
        #checks if data is type string
        if all(isinstance(value, str) for value in input_df[columnsList[i]]):
            encode_data(columnsList[i],input_df)

#finds the better regression model and returns the predicted price based on that model
def find_regression(input_df:pd.DataFrame,test_df:pd.DataFrame):

    #initializes both models
    model1 = LinearRegression()
    model2 = LogisticRegression(solver='liblinear')
    columnsList = input_df.columns.values
    xvals = input_df[columnsList[1:]]
    y = input_df['price']
  
    #splitting data into training and testing with 20% data allocated for testing
    x_train, x_test, y_train, y_test = train_test_split(xvals, y, test_size=0.2, random_state=42)

    #fits the training data to both of the models
    model1.fit(x_train, y_train)
    model2.fit(x_train,y_train)

    #gets the predictions from both the models
    y_predict1 = model1.predict(x_test)
    y_predict2 = model2.predict(x_test)

    #storing mean squared error of each model
    mse1 = mean_squared_error(y_test, y_predict1)
    mse2 = mean_squared_error(y_test, y_predict2)

    #compares the mean squared errors within the models and selects the model with less error
    if mse1 < mse2:
        model1.fit(xvals, y)
        y_predict = model1.predict(test_df)
        print('Selecting linear regression model with the mean squared error as ' + str(mse1) + ',')
    else:
        model2.fit(xvals, y)
        y_predict = model2.predict(test_df)
        print('Selecting logistic regression model with the mean squared error as ' +str(mse2)+',')

    #prints the predicted pricing
    print(y_predict)
    return y_predict



main()