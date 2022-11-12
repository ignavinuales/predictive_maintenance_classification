import pickle
import streamlit as st
from imblearn.ensemble import BalancedBaggingClassifier
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

with open('BalBagging_Multiclass.pkl', 'rb') as bagg:
    bagging_model = pickle.load(bagg)

with open('robustscaler.pkl', 'rb') as files:
    robust_scaler = pickle.load(files)

with open('minmaxscaler.pkl', 'rb') as files:
    minmax_scaler = pickle.load(files)

def classify(y_pred):
    if y_pred == 0:
        return 'OK (no failure)'

    elif y_pred == 1:
        return 'Power failure'

    elif y_pred == 2:
        return 'Tool wear failure'

    elif y_pred == 3:
        return 'Overstrain failure'
    
    elif y_pred == 4:
        return 'Heat dissipation failure'

def main(): 
    st.title('Predictive Maintenance. Failure classification')
    st.sidebar.header('User Imput Parameters')

    def user_input_params():
        quality = st.sidebar.selectbox('Product quality', ['Low', 'Medium', 'High'] )
        air_temp = st.sidebar.slider('Air temperature', 295, 304, 297)
        process_temp = st.sidebar.slider('Process temperature', 305, 315, 307)
        rotational_speed = st.sidebar.slider('Rotational speed', 1100, 3000, 1500)
        torque = st.sidebar.slider('Torque', 0, 80, 10)
        tool_wear = st.sidebar.slider('Tool wear', 0, 270, 20)

        robust_scaling = robust_scaler.transform([[rotational_speed, torque]]).ravel()
        rotational_speed = robust_scaling[0]
        torque = robust_scaling[1]

        minmax_scaling = minmax_scaler.transform([[air_temp, process_temp, tool_wear]]).ravel()
        air_temp = minmax_scaling[0]
        process_temp = minmax_scaling[1]
        tool_wear = minmax_scaling[2]

        if quality == 'Low':
            quality_prod = 0
        elif quality == 'Medium':
            quality_prod = 1
        else:
            quality_prod = 2

        data = {'Air temperature': air_temp,
                'Process temperature': process_temp,
                'Rotational speed': rotational_speed,
                'Torque': torque,
                'Tool wear': tool_wear,
                'Product quality': quality_prod
        }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_params()
    option = ['Bagging Classifier']
    model = st.sidebar.selectbox('Select classifier model', option)

    st.subheader('User input parameters')
    st.subheader(model)
    st.write(df)


    if st.button('RUN MODEL'):
        if model == 'Bagging Classifier':
            st.success(classify(bagging_model.predict(df)))

if __name__ == '__main__':
    main()

