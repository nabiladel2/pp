import pandas as pd
# Web Stramlit Packages :
import streamlit as st
from streamlit_option_menu import option_menu  # Correct import for option_menu

# Printing Output In Streamlit :
from contextlib import contextmanager, redirect_stdout
from io import StringIO

from Operations1 import Data_Showing , Data_Info , Data_Description , Data_Cleaning , Data_Matrix
st.title("Customer Segmentation using Clustering")

    # Output in Web Streamlit :
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
                    

# Sidebar with Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="User's Input For Data Preparation",
        options=[
               "Show Data",
               "Data Information",
               "Data Descripsion",
               "Data Cleaning",
               "Matrix Form"
        ],
    )

# Responding to User Selection
if selected == "Show Data":
       Data_Showing()

if selected == "Data Information":
       Data_Info()

if selected == "Data Descripsion":
       Data_Description()

if selected == "Data Cleaning":
       Data_Cleaning()

if selected == "Matrix Form":
       Data_Matrix()