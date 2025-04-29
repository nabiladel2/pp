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
from Operations1 import Histogram_Complain,Histogram_Income,Histogram_NumCatalogPurchases,Histogram_NumDealsPurchases
from Operations1 import Histogram_NumStorePurchases,Histogram_NumWebPurchases,Histogram_NumWebVisitsMonth,Histogram_Response

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
              "Histogram Income",
              "Histogram NumDealsPurchases",
              "Histogram NumWebPurchases",
              "Histogram NumCatalogPurchases",
              "Histogram NumStorePurchases",
              "Histogram NumWebVisitsMonth",
              "Histogram Complain",
              "Histogram Response"
        ],
    )


if selected_1 == "Histogram For Income" :
      Histogram_Income ()             

if selected_1 == "Histogram For NumDealsPurchases" :
      Histogram_NumDealsPurchases()

if selected_1 == "Histogram For NumWebPurchase " :
      Histogram_NumWebPurchases()

if selected_1 == "Histogram For NumCatalogPurchases" :
      Histogram_NumCatalogPurchases()

if selected_1 == "Histogram For NumStorePurchases" :
      Histogram_NumStorePurchases()

if selected_1 == "Histogram For NumWebVisitsMonth" :
      Histogram_NumWebVisitsMonth()
    
if selected_1 == "Histogram For Complain" :
      Histogram_Complain()

if selected_1 == "Histogram For Response" :
      Histogram_Response()