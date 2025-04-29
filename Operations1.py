import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import streamlit as st
from contextlib import contextmanager, redirect_stdout
from io import StringIO

# Streamlit Capture for Dynamic Output:
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


def Data_Read ():
    global data
    data = pd.read_csv("customer_segmentation.csv")
    data = pd.DataFrame(data)
   
def Data_Showing ():
    global data
    data = pd.read_csv("customer_segmentation.csv")
    data = pd.DataFrame(data)
    st.dataframe(data)
    st.write("Sum Of Missing Values")
    output = st.empty()
    with st_capture(output.code):
        print(data.isnull().sum().sum())
    
def Data_Info ():
    st.write("Data Information :")
    output = st.empty()
    with st_capture(output.code):
      print(data.info())

def Data_Description ():
    st.write("Data Description :")
    output = st.empty()
    with st_capture(output.code):
      print(data.describe(include ="all"))

def Data_Cleaning ():
    
    global New_Data
    st.write("Data After Cleaning")
    # Removing Missing Values And Duplicates :
    New_Data = data.dropna()
    New_Data.drop_duplicates()
    New_Data = pd.DataFrame(New_Data)
    st.dataframe(New_Data)

    st.write("Checking Missing Values After Cleaning :")
    output = st.empty()
    with st_capture(output.code):
        print(New_Data.isnull().sum())
    
    st.write("Sum Of Missing Values :")
    with st_capture(output.code):
         print(New_Data.isnull().sum().sum())   

    #New_Data.drop(columns=['Education', 'Marital_Status', 'ID','Dt_Customer'], inplace=True)              

def Data_Matrix ():
    corrmat= New_Data.corr()
    plt.figure(figsize=(20,20))  
    sns.heatmap(corrmat,annot=True,center=0)   

    global data
    data = pd.read_csv("customer_segmentation.csv")
    data = pd.DataFrame(data)
    New_Data = data.dropna()
    New_Data.drop_duplicates()
    New_Data = New_Data.drop(columns=['Education', 'Marital_Status', 'ID', 'Dt_Customer'])
    corrmat = New_Data.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corrmat, annot=True, center=0, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Boxplot For Some Features:

def Boxplot_Income ():
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("Income BoxPlot")
        # Customize boxplot appearance
        Boxplot_Income = sns.boxplot(New_Data['Income'])
        fig = px.box(New_Data['Income'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['Income']) )

def Boxplot_Recency ():
        
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("Recency BoxPlot")
        # Customize boxplot appearance
        Postal_Code_BoxPlot = sns.boxplot(New_Data['Recency'])
        fig = px.box(New_Data['Recency'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['Recency']))

def Boxplot_MntWines ():
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("MntWines BoxPlot")
        # Customize boxplot appearance
        Postal_Code_BoxPlot = sns.boxplot(New_Data['MntWines'])
        fig = px.box(New_Data['MntWines'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['MntWines']))

def Boxplot_MntFruits ():
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("MntFruits BoxPlot")
        # Customize boxplot appearance
        Postal_Code_BoxPlot = sns.boxplot(New_Data['MntFruits'])
        fig = px.box(New_Data['MntFruits'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['MntFruits']))

def Boxplot_MntMeatProducts ():
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("MntMeatProducts BoxPlot")
        # Customize boxplot appearance
        Postal_Code_BoxPlot = sns.boxplot(New_Data['MntMeatProducts'])
        fig = px.box(New_Data['MntMeatProducts'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['MntMeatProducts']))

def Boxplot_MntFishProducts ():
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("MntFishProducts BoxPlot")
        # Customize boxplot appearance
        Postal_Code_BoxPlot = sns.boxplot(New_Data['MntFishProducts'])
        fig = px.box(New_Data['MntFishProducts'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['MntFishProducts']))

def Boxplot_MntSweetProducts():
        data = pd.read_csv("customer_segmentation.csv")
        New_Data = pd.DataFrame(data)
        New_Data = New_Data.dropna()
        New_Data = New_Data.drop_duplicates()
        New_Data = New_Data.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], axis=1)
        st.write("MntSweetProducts BoxPlot")
        # Customize boxplot appearance
        Postal_Code_BoxPlot = sns.boxplot(New_Data['MntSweetProducts'])
        fig = px.box(New_Data['MntSweetProducts'] , color_discrete_sequence=["#636EFA", "#EF553B"])
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        output = st.empty()
        with st_capture(output.code):
           print("Outliers : " , len(New_Data['MntSweetProducts']))
        
# Histograms:

def Histogram_Income ():

    plt.hist(New_Data['Income'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of Income")
    st.pyplot(plt)

def Histogram_NumDealsPurchases ():

    plt.hist(New_Data['NumDealsPurchases'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of NumDealsPurchases")
    st.pyplot(plt)

def Histogram_NumWebPurchases ():

    plt.hist(New_Data['NumWebPurchases'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of NumWebPurchases")
    st.pyplot(plt)

def Histogram_NumCatalogPurchases ():

    plt.hist(New_Data['NumCatalogPurchases'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of NumCatalogPurchases")
    st.pyplot(plt)

def Histogram_NumStorePurchases ():

    plt.hist(New_Data['NumStorePurchases'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of NumStorePurchases")
    st.pyplot(plt)

def Histogram_NumWebVisitsMonth ():

    plt.hist(New_Data['NumWebVisitsMonth'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of NumWebVisitsMonth")
    st.pyplot(plt)

def Histogram_Complain ():
    plt.hist(New_Data['Complain'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of Complain")
    st.pyplot(plt)               
    
def Histogram_Response ():

    plt.hist(New_Data['Response'], color="blue")
    plt.ylabel("Freq.")
    plt.xlabel("Price $")
    plt.title("Distribution of Response")
    st.pyplot(plt)               
    

