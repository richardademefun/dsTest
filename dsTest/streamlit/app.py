#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components

#explain is a custom file available in this repo
from explain import pdplot, perm_import, perm_import_plot, shapValue, lime_explain

from desc import code, code2, overview_desc, home_page, fixed_head, code3, code4, code5

import pandas as pd

#matlplot lib used to do graph functions
import matplotlib.pyplot as plt

import os
import stat
import random 
import numpy as np
from io import StringIO
from PIL import Image
from dsTest.streamlit.pages.stage1_page import page1
from dsTest.streamlit.pages.stage2_page import page2

def main():
    
    #removing files already uploaded
    st.set_page_config(layout='wide', page_icon='\xf0\x9f\xa7\x8a',
                   page_title='ExpainMymodel')
    
    html_txt1 = """
        ### Currently only sklearn models are compatible
    """
    html_txt2 = """<font color='blue'>Upload files to Explain</font>"""

    hide_streamlit_style = \
        '''
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
    '''
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Disable warnings

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Select dashboard view from sidebar

    option = st.sidebar.selectbox('Select view', ('Stage1 Analysis', 'Stage 2 Modeling'))

    # Option to select different view of the app
            

    # if selected Tutorial option
    if option == 'Stage1 Analysis':
        page1()
    elif option == 'Stage 2 Modeling':
        page2()

if __name__ == '__main__':
    main()
