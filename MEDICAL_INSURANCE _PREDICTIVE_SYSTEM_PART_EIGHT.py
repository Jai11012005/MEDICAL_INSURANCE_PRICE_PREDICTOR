# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:07:15 2024

@author: user
"""

import numpy as np
import pickle
import streamlit as st

def medical_insurance_predictor(input_data, loaded_model):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction
    
def main():
    # Load the model
    loaded_model = pickle.load(open('C:/Users/user/Downloads/DEPLOYING MACHINE LEARNING MODEL/medical_insurance_cost_predictor.save', 'rb'))

    # Giving title
    st.title('MEDICAL INSURANCE PREDICTOR')
    
    age = st.number_input('PLEASE ENTER YOUR AGE HERE')
    sex = st.number_input('ENTER 1 FOR MALE AND 0 FOR FEMALE')
    bmi = st.number_input('ENTER YOUR BMI')
    children = st.number_input('ENTER THE NUMBER OF CHILDREN YOU HAVE', step=1)
    smoker = st.number_input('ENTER 1 IF YOU ARE A SMOKER ELSE ENTER 0')
    region = st.number_input('ENTER 0 FOR NORTHEAST, 1 FOR NORTHWEST, 2 FOR SOUTHEAST, AND 3 FOR SOUTHWEST', step=1)
    
    # Code for prediction
    charges = ''
    
    # Creating a predictive button
    if st.button('EXPECTED CHARGES'):
        charges = medical_insurance_predictor([age, sex, bmi, children, smoker, region], loaded_model)
        
    st.success(charges)

if __name__ == '__main__':
    main()

        
        
    
    
    
    