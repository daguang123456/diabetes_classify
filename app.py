import streamlit as st
import pandas as pd
import numpy as np
import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# from bokeh.plotting import figure

input_size = 8
hidden_size = 32
output_size = 1

class DiabetesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

@st.cache(suppress_st_warning=True)
def get_fvalue(val):    
	feature_dict = {"No":1,"Yes":2}    
	for key,value in feature_dict.items():        
		if val == key:            
			return value

def get_value(val,my_dict):    
	for key,value in my_dict.items():        
		if val == key:            
			return value
app_mode = st.sidebar.selectbox('选择页面',['主页','预测']) #two pages


if app_mode=='主页':    
	st.title('糖尿病:')      
	# st.image('loan_image.jpg')    
	st.markdown('数据:')    
	data=pd.read_csv('diabetes.csv')    
	st.write(data.head(15))    
	st.markdown('图表')
	# st.bokeh_chart(data[['Glucose','BloodPressure']].head(20))
	# fig = plt.plots()
	# st.bar_chart(data[['Glucose','BloodPressure']].head(20))
	# ax1 = data.plot.scatter(x='Glucose',
    #                   y='BMI')
	# plt.scatter(data.Glucose, data.BMI ) 
	# st.pyplot(fig)
	# st.bar_chart(data[['Glucose','BMI']].head(20))
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ax.scatter(data['Glucose'], data['BloodPressure'], color='r')
	# ax.scatter(grades_range, boys_grades, color='b')
	ax.set_xlabel('Glucose')
	ax.set_ylabel('Blood Pressure')
	ax.set_title('Glucose vs Blood Pressure')
	st.pyplot(fig)

	fig1=plt.figure()
	ax1=fig1.add_axes([0,0,1,1])
	ax1.scatter(data['Age'], data['BloodPressure'], color='r')
	# ax.scatter(grades_range, boys_grades, color='b')
	ax1.set_xlabel('Age')
	ax1.set_ylabel('Blood Pressure')
	ax1.set_title('Age vs Glucose')
	st.pyplot(fig1)

	fig2=plt.figure()
	ax2=fig2.add_axes([0,0,1,1])
	ax2.scatter(data['BloodPressure'], data['Outcome'], color='r')
	# ax.scatter(grades_range, boys_grades, color='b')
	ax2.set_xlabel('BloodPressure')
	ax2.set_ylabel('Outcome,1 indicating have diabetes')
	ax2.set_title('Blood Pressure vs Outcome')
	st.pyplot(fig2)

elif app_mode == '预测':
	# st.image('slider-short-3.jpg')    
	st.subheader('先生/女士，您需要填写所有必要的信息，以便得到对您的诊断请求的答复(模型精度--0.71)！')
	st.sidebar.header("Informations about the client :")    
	# gender_dict = {"Male":1,"Female":2}    
	# feature_dict = {"No":1,"Yes":2}    
	# edu={'Graduate':1,'Not Graduate':2}    
	# prop={'Rural':1,'Urban':2,'Semiurban':3}    
	preg=st.sidebar.number_input('怀孕次数',0,20,0,)    
	gluc=st.sidebar.number_input('葡萄糖(md/dl)',0.00,250.00,0.00,)    
	bloodpr=st.sidebar.number_input('舒张性血压(mm hg)', 0.00,150.00,0.00)   
	skinthic=st.sidebar.number_input('表皮厚度', 0.00,100.00,0.00) 
	insulin = st.sidebar.number_input('胰岛素', 0.00,850.00,0.00) 
	bmi = st.sidebar.number_input('暴模指数', 0.00,68.00,0.00) 
	diapedig = st.sidebar.number_input('糖尿病谱系功能', 0.00,3.00,0.00) 
	age = st.sidebar.number_input('年龄', 0.00,90.00,0.00) 

	# data1={    
	# 	'Gender':Gender,    
	# 	'Married':Married,    
	# 	'Dependents':[class_0,class_1,class_2,class_3],    
	# 	'Education':Education,    
	# 	'ApplicantIncome':ApplicantIncome,    
	# 	'CoapplicantIncome':CoapplicantIncome,    
	# 	'Self Employed':Self_Employed,    
	# 	'LoanAmount':LoanAmount,    
	# 	'Loan_Amount_Term':Loan_Amount_Term,    
	# 	'Credit_History':Credit_History,    
	# 	'Property_Area':[Rural,Urban,Semiurban],    
	# 	}    
	# feature_list=[
	# 	ApplicantIncome,
	# 	CoapplicantIncome,
	# 	LoanAmount,
	# 	Loan_Amount_Term,
	# 	Credit_History,
	# 	get_value(Gender,gender_dict),
	# 	get_fvalue(Married),
	# 	data1['Dependents'][0],
	# 	data1['Dependents'][1],
	# 	data1['Dependents'][2],
	# 	data1['Dependents'][3],
	# 	get_value(Education,edu),
	# 	get_fvalue(Self_Employed),
	# 	data1['Property_Area'][0],
	# 	data1['Property_Area'][1],
	# 	data1['Property_Area'][2]
	#  	]    
	# single_sample = np.array(feature_list).reshape(1,-1)
	feature_list=[
		preg,
		gluc,
		bloodpr,
		skinthic,
		insulin,
		bmi,
		diapedig,
		age
	 	]    
	# single_sample = np.array(feature_list).reshape(1,-1)

	if st.button("预测"):        
		# file_ = open("6m-rain.gif", "rb")        
		# contents = file_.read()        
		# data_url = base64.b64encode(contents).decode("utf-8")        
		# file_.close()        
		# file = open("green-cola-no.gif", "rb")        
		# contents = file.read()        
		# data_url_no = base64.b64encode(contents).decode("utf-8")        
		# file.close()        
		# Load the model
		model = DiabetesModel(input_size, hidden_size, output_size)
		model.load_state_dict(torch.load("diabetes_model.pt"))
		# Set the model to evaluation mode
		model.eval()
		# loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))        
		# prediction = loaded_model.predict(single_sample)    
        # Make predictions on a sample input
		sample_input = torch.tensor([[feature_list]])
		with torch.no_grad():
		    predicted_output = model(sample_input.float())
		    # print(f"Predicted output: {predicted_output.item()}")
		    st.success( f"患糖尿病的概率：: {predicted_output.item()}"  )

		# if prediction[0] == 0 :            
		# 	st.error(    'According to our Calculations, you will not get the loan from Bank'    )            
		# 	# st.markdown(    f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">',    unsafe_allow_html=True,)        
		# elif prediction[0] == 1 :            
		# 	st.success(    'Congratulations!! you will get the loan from Bank'    )            
		# 	st.markdown(    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',    unsafe_allow_html=True,    )
	


