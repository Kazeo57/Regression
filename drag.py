import streamlit as st
import joblib
import pandas as pd 

def predict_insurance(new_data):
    #new_data=pd.read_excel()
    best_model=joblib.load('best_insurance.pkl')
    scalers=joblib.load('scalers.pkl')
    region_encoder=joblib.load("region_encoder.pkl")
    new_data['sex'].replace(['male','female'],[0,1],inplace=True)
    new_data['smoker'].replace(['no','yes'],[0,1],inplace=True)
    new_data["region"]=region_encoder.transform(new_data[["region"]])
    new_data.head()
    print(scalers)
    #st.write("Yo yo")
    for i in new_data.columns:
        if i in scalers:
            print(i)
            scaler=scalers[f"{i}"]
            print(scaler)
            new_data[[f"{i}"]]=scaler.transform(new_data[[f"{i}"]])
            #new_data.head()
    
    
    charges=best_model.predict(new_data)
    return charges
    #new_data.head()

def handle_prediction():
    if 'data_loaded' in st.session_state:
        st.session_state['prediction']=predict_insurance(st.session_state['data_loaded'])
        #st.success("Prediction made sucessfully")

st.header("Upload your Table here")
upload_file=st.file_uploader("Choose your CSV",type="csv")

#upload_file="Insurance-data.xlsx"

if upload_file:
    df=pd.read_csv(upload_file)
    st.session_state['data_loaded']=df
    st.dataframe(df)
    st.write("PREDICT NOW ...")
    st.button("Predict",on_click=handle_prediction)

    #st.write(st.session_state)    
if 'prediction' in st.session_state:
    prediction=st.session_state['prediction']
    #st.write(prediction)
    st.success("Prediction made sucessfully")
    predictions=pd.DataFrame(prediction,columns=["Predictions"])
    st.write(predictions)
    #predictions=predictions.reset_index(drop=True)
    predictions.to_excel("Charges.xlsx")
    result=pd.concat([df,predictions],axis=1)
    st.dataframe(result)
    #result.to_csv("Your_Prédictions",index=False)
    #st.success("Great")
    #else:
    #    print("Casalaba!!!!")

    #st.write('Ya')

    #Télécharger sous plusieurs format 
    #st.download_button(
    #    label='Excel',
    #    data=result.to_excel(index=False),
    #    file_name='Predict.xlsx',
    #    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    #)

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")

    csv = convert_df(result)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="Result.csv",
        mime="text/csv",
    )

