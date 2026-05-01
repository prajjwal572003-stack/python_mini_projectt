from main import Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error ,r2_score
from sklearn.model_selection import train_test_split

def make_predictions(inputs: list[float], outputs: list[float], input_value , plot: bool= False)->Prediction:
    if len(inputs)!=len(outputs):
        raise Exception("length of input must match with the length of output")
    
    #createing a dataframe for our data
    df=pd.DataFrame({'inputs':inputs,'outputs':outputs})

    X=np.array(df['inputs']).reshape(-1,1)
    y=np.array(df['outputs']).reshape(-1,1)

    print(X)
    print(y)

    #split the data into training and testing data
    train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=0,test_size=0.20)


    #initialize the model
    model=LinearRegression()
    model.fit(train_X,train_y)

    #prediction
    y_prediction=model.predict([[input_value]])
    y_line=model.predict(X)


    #testing the accuraccy
    y_test_prediction=model.predict(test_X)

    if plot:
        display_plot(inputs=X,outputs=y,y_line=y_line)
        

    return Prediction(value=y_prediction[0][0],r2_score=r2_score(test_y,y_test_prediction),slope=model.coef_[0][0],intercept=model.intercept_[0],mean_absolute_error=mean_absolute_error(test_y,y_test_prediction))


def display_plot(inputs: list[float],outputs:list[float],y_line):
    plt.scatter(inputs,outputs,s=12)
    plt.xlabel('Inputs')
    plt.ylabel('outputs')
    plt.plot(inputs,y_line,color='r')
    plt.show()

if __name__=="__main__":
   years: list[int]=[1,2,3,4,5,6]
   earnings:list[int]=[1000,800,2000,1500,3400,4000]
   my_input: int=10

   prediction = make_predictions(inputs=years, outputs=earnings, input_value=my_input, plot=True)