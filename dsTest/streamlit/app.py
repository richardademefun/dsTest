import streamlit as st

from dsTest.streamlit.pages.stage1_page import page1
from dsTest.streamlit.pages.stage2_page import page2


def main():
    st.set_page_config(layout='wide', page_icon='\xf0\x9f\xa7\x8a',
                       page_title='DS Test')

    hide_streamlit_style = \
        '''
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
    '''
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    option = st.sidebar.selectbox('Select view', ('Stage1 Analysis', 'Stage 2 Modeling'))

    if option == 'Stage1 Analysis':
        page1()
    elif option == 'Stage 2 Modeling':
        page2()


if __name__ == '__main__':
    main()
