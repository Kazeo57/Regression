import pandas as pd
import streamlit as st
from PIL import Image
#import plotly.express as px
import joblib 
import os
#new_data=pd.DataFrame({"age":age,"sex":sex,"bmi":bmi,"children":children,"smoker":smoker,"region":region})
def predict_insurance(age,sex,bmi,children,smoker,region):
    new_data=pd.DataFrame({"age":[age],"sex":[sex],"bmi":[bmi],"children":[children],"smoker":[smoker],"region":[region]})
    #best_model=joblib.load('best_insurance.pkl')
    best_model=joblib.load(os.path.join(os.path.dirname(__file__), 'best_insurance.pkl'))
    #scalers=joblib.load('scalers.pkl')
    scalers=joblib.load(os.path.join(os.path.dirname(__file__), 'scalers.pkl'))
    #region_encoder=joblib.load("region_encoder.pkl")
    region_encoder=joblib.load(os.path.join(os.path.dirname(__file__), 'region_encoder.pkl'))
    new_data['sex'].replace(['male','female'],[0,1],inplace=True)
    new_data['smoker'].replace(['no','yes'],[0,1],inplace=True)
    new_data["region"]=region_encoder.transform(new_data[["region"]])
    new_data.head()
    
    for i in new_data.columns:
        if i in scalers:
            #print(i)
            scaler=scalers[f"{i}"]
            #print(scaler)
            new_data[[f"{i}"]]=scaler.transform(new_data[[f"{i}"]])
            
    
    
    charges=best_model.predict(new_data)
    return charges



#age	sex	bmi	children	smoker	region	charges
with st.form(key="form", clear_on_submit=False, border=True):
    st.write("Attribute")
    #Age
    age=st.slider("Age")
    #Sex
    st.write("Sex")
    #sex=st.checkbox("Male"),st.checkbox("Female")
    sex=st.radio(
        "What's your gender?",
        ["male","female"]
    )
    #bmi
    bmi=st.number_input("Bmi")
    #Children
    num_children=st.slider("Number of Children",max_value=20)
    #Smoker
    smoker=st.checkbox("Smoker")
    #Region
    region=st.selectbox("Where are you living?",("southeast","northeast","southwest","nortwest"))


    submitted=st.form_submit_button(label="Submit",help=None,on_click=None,args=None,kwargs=None,type="secondary",disabled=False,use_container_width=False)
    charges=predict_insurance(age,sex,bmi,num_children,smoker,region)
    if submitted:
        st.write("Your insurance charge is :",charges)
        

st.write("Outside the form")