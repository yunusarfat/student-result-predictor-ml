import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('student_rf_pipeline.pkl', 'rb') as file:
    rf_model = pickle.load(file)
#main logic 
def predict_gpa(gender,age, address, famsize,pstatus, M_Edu,
                F_Edu, M_Job,F_Job,relationship,smoker,tuition_fee,time_friends,ssc_result):
    input_df=pd.DataFrame([[
        gender,age, address, famsize,pstatus, M_Edu,
        F_Edu, M_Job,F_Job,relationship,smoker,tuition_fee,time_friends,ssc_result
    ]],
    columns=[
        'gender', 'age', 'address', 'famsize', 'Pstatus', 'M_Edu',
        'F_Edu', 'M_Job', 'F_Job', 'relationship','smoker', 'tuition_fee', 'time_friends', 'ssc_result'  

    ])
    # prediction
    prediction = rf_model.predict(input_df)[0]
    # clipping
    return np.clip(prediction, 0, 5)

inputs=[
    gr.Radio(["M", "F"], label="Gender"),
    gr.Number(label="Age",value=18),
    gr.Radio(["Urban", "Rural"], label="Address"),
    gr.Radio(["LE3", "GT3"], label="Family Size"),
    gr.Radio(["T", "A"], label="Parental Status"),
    gr.Slider(0,4,step=1,label="Mother's Education"),
    gr.Slider(0,4,step=1,label="Father's Education"),
    gr.Dropdown(["At_home","Health","other","services","Teacher"],
                    label="Mother's Job"),
    gr.Dropdown(["Business","Health","other","services","Teacher","Farmer"],  
                    label="Father's Job"),

    gr.Radio(["Yes","No"], label="Smoker"),
    gr.Radio(["Yes","No"], label="Relationship"),
    gr.Number(label="Tuition Fee",value=0),
    gr.Slider(1,5 , step=1,label="Time with Friends"),
    gr.Number(label="SSC Result",value=0)

]

#gradio interface
app=gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs="text",
    title="HSE Result Predictor",
)

#launch gradio interface
app.launch(share=True)
