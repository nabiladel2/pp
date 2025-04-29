import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_option_menu import option_menu  
import plotly.express as px
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from Operations import Boxplot_Income,Boxplot_Recency,Boxplot_MntFishProducts,Boxplot_MntFruits,Boxplot_MntMeatProducts,Boxplot_MntSweetProducts,Boxplot_MntWines


def app():
    st.title("Data Outliers")

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



# Display Title
st.title("Data Outliers Detections")

with st.sidebar:
    selected_1 = option_menu(
        menu_title="Data Outliers Detections",
        options=[
            "Boxplot For  Income",
            "Boxplot For Recency",
            "Boxplot For MntWines",
            "Boxplot For MntFruits",
            "Boxplot For MntMeatProducts",
            "Boxplot For MntFishProducts",
            "Boxplot For MntSweetProducts"
        ],
    )


if selected_1 == "Boxplot For Income" :
      Boxplot_Income()             

if selected_1 == "Boxplot For Recency" :
      Boxplot_Recency()

if selected_1 == "Boxplot For MntWines" :
      Boxplot_MntWines()

if selected_1 == "Boxplot For MntFruits" :
      Boxplot_MntFruits()

if selected_1 == "Boxplot For MntMeatProducts" :
      Boxplot_MntMeatProducts()

if selected_1 == "Boxplot For MntFishProducts" :
      Boxplot_MntFishProducts()
    
if selected_1 == "Boxplot For MntSweetProducts" :
      Boxplot_MntSweetProducts()