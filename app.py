import pandas as pd
import numpy as np

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode()
import plotly.express as px
import pickle
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings
with warnings.catch_warnings():
# ignore all caught warnings
    warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Code
diabetes_data = pd.read_csv('diabetes.csv')
Non_Diabetic = diabetes_data[diabetes_data['Outcome'] == 0]
Diabetic = diabetes_data[diabetes_data['Outcome'] == 1]
def plot_distribution(data_select, size_bin) :  
    # 2 datasets
    tmp1 = Diabetic[data_select]
    tmp2 = Non_Diabetic[data_select]
    hist_data = [tmp1, tmp2]
    group_labels = ['Diabetic', 'Non-Diabetic']
    colors = ['Red', 'Green']
    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    fig['layout'].update(title = data_select)
    st.write(fig)

Non_Diabetic= Non_Diabetic.sample(n=268)
final_df = pd.concat([Non_Diabetic,Diabetic], axis=0)
x0 = np.array(diabetes_data['Pregnancies'])
x1 = np.array(diabetes_data['Glucose'])
x2 = np.array(diabetes_data['BloodPressure'])
x3 = np.array(diabetes_data['SkinThickness'])
x4 = np.array(diabetes_data['Insulin'])
x5 = np.array(diabetes_data['BMI'])
x6 = np.array(diabetes_data['DiabetesPedigreeFunction'])
x7 = np.array(diabetes_data['Age'])
x8 = np.array(diabetes_data['Outcome'])
figb = go.Figure()
figb.add_trace(go.Box(x=x0,name='Pregnancies'))
figb.add_trace(go.Box(x=x1,name='Glucose'))
figb.add_trace(go.Box(x=x2,name='BloodPressure'))
figb.add_trace(go.Box(x=x3,name='SkinThickness'))
figb.add_trace(go.Box(x=x4,name='Insulin'))
figb.add_trace(go.Box(x=x5,name='BMI'))
figb.add_trace(go.Box(x=x6,name='DiabetesPedigreeFunction'))
figb.add_trace(go.Box(x=x7,name='Age'))
figb.add_trace(go.Box(x=x8,name='Outcome'))
figb.update_layout(height=600, width=800, title_text="Boxplots")


X_df = diabetes_data.drop(["Outcome"],axis=1)
y_df= diabetes_data['Outcome']

sm = SMOTE(random_state=24)
X,y = sm.fit_resample(X_df, y_df)

x_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(x_scaled,y,test_size = 0.20)

pickle_in = open('gbc.pkl', 'rb')
gbc = pickle.load(pickle_in)

def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age):
    prediction = gbc.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age]])
    return prediction

# this is the main function in which we define our webpage
def main():
    html_temp = """
    <div style="background-color:#f63366 ;padding:10px;margin-bottom:10px;">
    <h1 style="color:white;text-align:center;">Diabetes Prediction Web App</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.title("Pages")
    pages=['About Dataset','Data Preprocessing','Exploratory Data Analysis', 'Model Training' , 'Model Comparisons','Predictions']
    add_pages = st.sidebar.selectbox('', pages)

    if add_pages=='About Dataset':
        html_temp2 = """
        <body >
        <h3>About Dataset</h3>
        This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
        The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
        </body>
        """
        st.markdown(html_temp2, unsafe_allow_html=True)
        #Loading the dataset
        st.header("Dataset")
        st.subheader("First 5 Rows of the dataset")
        st.write(diabetes_data.head())
        st.subheader("Last 5 Rows of the dataset")
        st.write(diabetes_data.tail())
        st.write("The number of rows and columns:", diabetes_data.shape)
        st.subheader("Dataset Description")
        st.write(diabetes_data.describe())
        st.header("Target Variable")
        st.write('Non-Diabetic Cases: {}'.format(len(Non_Diabetic)))
        st.write('Diabetic Cases: {}'.format(len(Diabetic)))
        trace = go.Pie(labels = ['Non_diabetic','diabetic'], values = diabetes_data['Outcome'].value_counts(), 
                   textfont=dict(size=15), opacity = 0.8,
                   marker=dict(colors=['green', 'red'], 
                               line=dict(color='#000000', width=1.5)))
        layout = dict(title =  'Distribution of Outcome variable')
        fig = dict(data = [trace], layout=layout)
        st.write(fig)
        # 2 datasets
    if add_pages=='Data Preprocessing':
        st.header("Missing Values")
        diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
        st.write(diabetes_data.isna().sum())

        col=['Glucose' ,'BloodPressure' ,'SkinThickness', 'Insulin' ,'BMI']
        for i in col:
            diabetes_data[i].replace(np.NaN,diabetes_data[i].mean(),inplace=True)

        st.header("Box Plots")
        st.write(figb)
        st.header("Correlation Matrix")
        # Correlation matrix
        corrmat = diabetes_data.corr()
        fig = go.Figure(data = go.Heatmap( z = corrmat.values, x = list(corrmat.columns),y = list(corrmat.index),colorscale = 'Viridis'))
        fig.update_layout(title = 'Correlation')
        st.write(fig)
        #Correlation with output variable
        cor_target = abs(corrmat["Outcome"])
        #Selecting highly correlated features
        relevant_features = cor_target[cor_target>0.1]
        st.write(relevant_features)

    if add_pages=='Exploratory Data Analysis':
        st.header("Data Distribution")
        plot_distribution('Insulin', 0)
        plot_distribution('Glucose', 0)
        plot_distribution('BloodPressure', 5)
        plot_distribution('BMI', 0)
        plot_distribution('Age', 0)
        plot_distribution('Pregnancies', 0)
        plot_distribution('DiabetesPedigreeFunction', 0)
        st.header("Scatterplot Matrix")
        fig = px.scatter_matrix(diabetes_data,dimensions=diabetes_data.columns,
            color="Outcome")
        fig.update_layout(
            title='Scatterplot Matrix',
            dragmode='select',
            width=1300,
            height= 1300,
            hovermode='closest',
        )
        st.write(fig)

    if add_pages=='Model Training':
        st.header("SMOTE(synthetic minority oversampling technique)")
        st.subheader("Actual Dataset")
        st.write(X_df.shape)
        st.write(y_df.shape)
        st.subheader("After Oversampling")
        st.write(X.shape)
        st.write(y.shape)
        st.header("Standard Scaler")
        st.subheader("Before Scaling")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=x0,name='Pregnancies'))
        fig.add_trace(go.Histogram(x=x1,name='Glucose'))
        fig.add_trace(go.Histogram(x=x2,name='BloodPressure'))
        fig.add_trace(go.Histogram(x=x3,name='SkinThickness'))
        fig.add_trace(go.Histogram(x=x4,name='Insulin'))
        fig.add_trace(go.Histogram(x=x5,name='BMI'))
        fig.add_trace(go.Histogram(x=x6,name='DiabetesPedigreeFunction'))
        fig.add_trace(go.Histogram(x=x7,name='Age'))
        fig.update_layout(height=800, width=800, title_text="Histograms")
        st.write(fig)
        st.subheader("After Scaling")
        st.write(x_scaled)
        st.header("Splitting Dataset")
        st.write(X_train.shape)
        st.write(X_test.shape)
        st.write(y_train.shape)
        st.write(y_test.shape)
        st.header("Logistic Regression")
        st.write("Train Set Accuracy:74.75")
        st.write("Test Set Accuracy:78.5" )
        st.header("KNN Classifier")
        st.write("Train Set Accuracy:80.75")
        st.write("Test Set Accuracy:79.5" )
        st.header("SVM Claasifier")
        st.write("Train Set Accuracy:74.5")
        st.write("Test Set Accuracy:78.0" )
        st.header("Decision Tree Classifier")
        st.write("Train Set Accuracy:100")
        st.write("Test Set Accuracy:77" )
        st.header("Gradient Boosting Classifier")
        st.write("Train Set Accuracy:91.5")
        st.write("Test Set Accuracy:85.0" )
        st.header("XGB Classifier")
        st.write("Train Set Accuracy:88.875")
        st.write("Test Set Accuracy:79.5" )

    if add_pages == 'Model Comparisons':
        st.header("Model Comparisons")
        models = pd.DataFrame({
        'Model': ['Logistic','KNN', 'SVC',  'Decision Tree Classifier',
                'Gradient Boosting Classifier',  'XgBoost'],
        'Score': [ 0.785, 0.795,0.780, 0.770, 0.850,0.795]
    })
        models.sort_values(by = 'Score', ascending = False)
        colors=['Logistic','KNN', 'SVC',  'Decision Tree Classifier',
             'Gradient Boosting Classifier',  'XgBoost']
        fig = px.bar(models, x='Model', y='Score',color=colors)
        st.write(fig)

    if add_pages=="Predictions":
        Pregnancies = st.number_input("Pregnancies",min_value=0, max_value=6000000, step=1,format="%i")
        Glucose = st.number_input("Glucose",min_value=0, max_value=6000000, step=1,format="%i")
        BloodPressure = st.number_input("Blood Pressure",min_value=0, max_value=6000000, step=1,format="%i")
        SkinThickness = st.number_input("Skin Thickness",min_value=0, max_value=6000000, step=1,format="%i")
        Insulin  = st.number_input("Insulin",min_value=0, max_value=6000000, step=1,format="%i")
        BMI = st.number_input("BMI",min_value=0, max_value=6000000, step=1,format="%i")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function",min_value=0, max_value=6000000, step=1,format="%i")
        Age = st.number_input("Age",min_value=0, max_value=6000000, step=1,format="%i")
        result =""
        user_report_data = {'Pregnancies':Pregnancies, 'Glucose': Glucose,'BloodPressure': BloodPressure, 'SkinThickness':SkinThickness, 'Insulin':Insulin,'BMI':BMI, 'DiabetesPedigreeFunction':DiabetesPedigreeFunction, 'Age':Age}
        report_data = pd.DataFrame(user_report_data, index=[0])
        st.subheader('User Input Data')
        st.write(report_data)
        # the below line ensures that when the button called 'Predict' is clicked,
        # the prediction function defined above is called to make the prediction
        # and store it in the variable result
        st.subheader('Result: ')
        if st.button("Predict"):
            result = prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age)
            if result == 0:
                st.success('The Patient is Not Diabetic')
            if result == 1:
                st.failure('The Patient is Diabetic')


if __name__=='__main__':
    main()