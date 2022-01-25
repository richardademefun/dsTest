import streamlit as st
from PIL import Image
from dsTest.constants import initial_df, common_ids, reg_CPA_distribution, dep_CPA_distribution, ctr_data, ctr_result
from dsTest.streamlit.desc import code, code2, overview_desc, fixed_head, code3, code4, code5


def page1():
    st.sidebar.markdown(overview_desc)
    st.write(fixed_head, unsafe_allow_html=True)
    st.write('## Lets have a look at the results data')
    st.code(code, language='python')
    image = Image.open(initial_df)

    st.image(image, caption='initial_df')

    st.write("""From a first look at the data we can see that in this one set of results is 2 separate datasets. 
    One dataset to Do the CPA analysis and the Click Through rate. You can see this as when ever pixel id != 0 
    there are no ctr related metrics for example.""")

    st.code(code2, language='python')
    image = Image.open(common_ids)
    st.image(image, caption='common ids')
    st.write('## CPA Analysis')

    st.write("""Since I was given th choice as to which Line_item_id i get to analysed. I have chosen the
    one with the highest occurrences of our target KPI's(Target Registration CPA: $1,000 (Pixel ID: 1433481) &
    Target Deposit CPA: $1,400 (Pixel ID: 1433583))""")

    st.code(code3, language='python')
    st.code(code4, language='python')

    st.write("Drop all irrelevant columns. And create classifying column for both desired CPA value of 1433583 and "
             "1433481")

    st.code(code5, language='python')

    st.write("""Aggregate the model to the model leaf to evaluate how well models are performing on average. Moreover 
    we need to also know what a good value for form completion an first deposit value would be. I found a link to an 
    article explaining that the average sign up rate is [30%](https://baymard.com/lists/cart-abandonment-rate). 
    Unfortunately I couldn't find anything similar for first deposit rates so i made the assumption that 1/3 of 
    people who complete the sign up also make a deposit giving me an average rate of 10% for the Target Deposit CPA 
    value.""")

    image = Image.open(reg_CPA_distribution)
    st.image(image, caption='Target Registration CPA')

    image = Image.open(dep_CPA_distribution)
    st.image(image, caption='Target Deposit CPA')

    st.write("""For Target Registration CPA the best model we have gets a weight of 0.95 (RT00031).
    For Target Deposit CPA we have a model with max rate of 10 (9RT00088). I also made a hybrid
    score that combines the signup and deposit rates using the value as a weight. This shows that 
    the models of RT00154 gets a weight of 5.8 based on thi hybrid score. I may be possible to 
    further see the model that performs best by value using rpa_adv_curr field. however i'm not 100% 
    sure what this field means.""")

    st.write('## CTR Analysis')

    st.write("""For the ctr analysis we went through a similar process as above where we filtered the 
    data to keep all the fields and rows relevant to the ctr. Moreover when checking for the frequency
    of non zero ctr again we see that a custom_model_id of 44869469 is the the second best in terms 
    of frequency.
    """)

    image = Image.open(ctr_data)
    st.image(image, caption='Target Deposit CPA')

    st.write("""For Ctr i also aggregated it using the mean. i also tried a using a weighted mean aggregation 
    which weights ctr higher with less impressions. Furthermore i calculated a weight based on the impression
    weighted mean and the rpm value.
    """)
    image = Image.open(ctr_result)
    st.image(image, caption='Target Deposit CPA')
