
fixed_head = """
    <h1 style="color:#d1ab69;text-align:center;letter-spacing:4px;font-family:"Times New Roman", Times, serif;" > DS Tech Test</h1>
"""

home_page = f"""
    
        <div style="display:flex;justify-content:space-between;background:#01203b;padding:10px;border-radius:5px;margin:10px;">
            <div style="float:right;width:100%;background:#01203b;padding:10px;border-radius:5px;margin:10px;">
                <h2 style="color:#d1ab69;text-align:center;letter-spacing:4px;font-family:"Times New Roman", Times, serif;" >1. Custom Algorithm Performance - CPS Analysis</h2>
                    <h6 style="color:white;letter-spacing:1px;line-height: 1.6;font-family:Arial, Helvetica, sans-serif;">
                        Stage 1 requires us to invstigate and provide a recommended action for each custom_model_leaf_name in the form of a bid multiplier
                        where a multiplier of less than 1 reduces our bid, and a multiplier of more than 1 increases our bid. To do this we'll
                        have to check for any preprocessing. We'll have to choose one line_item_id and of corse explain all of our choices.
                    </6>
                <br><br>
                </2>
                <h4 style="color:white;letter-spacing:1px;line-height: 1.6;font-family:Arial, Helvetica, sans-serif;">
                    2. Having a look at the data
                </h4>
            </div>
        </div>

"""

code = """import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import numpy as np 
algo_performance_loc = "./test_algo_performance.csv"
algo_performance = pd.read_csv(algo_performance_loc)
algo_performance
"""

code2 = """
CPA_data = algo_performance[algo_performance['pixel_id'] != 0]
CPA_data[(CPA_data['pixel_id'] == 1433583) | (CPA_data['pixel_id'] == 1433581)]['custom_model_id'].value_counts()
"""

code3 = """
data_set.drop(
    ['advertiser_id', 
     'insertion_order_id', 
     'line_item_id', 
     'impressions', 
     'clicks',
     'custom_model_id',
     'booked_revenue_adv_curr', 
     'booked_revenue',
     'ctr',
     'rpm'
    ], 
    axis=1, 
    inplace=True
)
"""

code4 = """
data_set['TD_CPA'] = (data_set['pixel_id'] == 1433583)*1
data_set['TR_CPA'] = (data_set['pixel_id'] == 1433581)*1
"""

code5 = """
agg_data = data_set.groupby('custom_model_leaf_name').agg({'conversions': 'mean', 
                                                'rpa_adv_curr': 'mean', 
                                                'TD_CPA': 'mean', 
                                                'TR_CPA': 'mean'})
"""

overview_desc = """
    
    ------------------------
    ## Overview

    #### I'm trying out Streamlit because it looks cool. 
    -------------------------
    ## Instructions

    1. Github [Repo](https://github.com/richardademefun/dsTest)

    2. Clone the repo

    3. GO!
    -------------------------------

     """