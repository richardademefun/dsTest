import streamlit as st
from desc import fixed_head
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet, PoissonRegressor, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from dsTest.constants import employment_data_path, demographic_data_path, sample_param_data

def page2():

    st.write(fixed_head, unsafe_allow_html=True)

    html_txt1 = """
        ### Currently only sklearn models are compatible
    """
    html_txt2 = """<font color='blue'>Upload files to Explain</font>"""

    data = pd.read_csv(sample_param_data, index_col=0)

    data_path = st.sidebar.file_uploader('Input file', type=['csv', 'text'])
    if data_path is not None:
        data = pd.read_csv(data_path, index_col=0)

    st.write("""Similarly to the first stage I chose a sample of line_item_id to try to model (13917115).
    Again we need to split the data into 2 ssection for the cpa and ctr modelling. See sample df below.
    """)
    st.dataframe(data.iloc[:5])

    st.write("""The most obious and troublsom feature of this data is the distribution of of ctr. 
    The data set is very heavy on zeros Meaning that we can't just model it as it is using any method
    Really. There for we're going to aggrigate the data and try to classsify "obvious" 
    instances of zero ctr. i'm all going to talk a little bit about adding demographic data and why
    that could furtehr improve the model.""")

    ctr_data = data[data['pixel_id'] == 0]
    ctr_data[ctr_data['line_item_id'] == 13917115].groupby(
            "postal_code").agg({"ctr": ['mean', 'count']})['ctr'].sort_values(by='mean')

    ctr_data_cluster = ctr_data.drop(
    ['advertiser_id', 'insertion_order_id', 'line_item_id', 'region', 'city', 'pixel_id', 
    'impressions', 'clicks', 'booked_revenue_adv_curr', 'booked_revenue', 'conversions', 
    'rpm', 'rpa_adv_curr', 'country'], axis=1)

    categorical_variables = ['device_type', 'weekday_user_tz', 'hour_user_tz']
    ctr_data_cluster[categorical_variables] = ctr_data_cluster[categorical_variables].astype(object)

    one_hot = pd.get_dummies(ctr_data_cluster[categorical_variables])#, drop_first=True)
    ctr_data_cluster = ctr_data_cluster.drop(categorical_variables,axis = 1)
    ctr_data_cluster = ctr_data_cluster.join(one_hot)

    ctr_count = ctr_data_cluster.groupby(
            "postal_code").agg(
            {'postal_code':'count', 
            'dma': 'first'})

    ctr_agg = ctr_data_cluster.groupby(
            "postal_code").agg(
            'sum')

    ctr_agg['dma'] = ctr_count['dma']

    ctr_agg['count'] = ctr_count['postal_code']

    for col in ctr_agg.columns:
        if col != 'dma':
            ctr_agg[col] = ctr_agg[col] / ctr_agg['count']

    ctr_agg.drop('count', inplace=True, axis=1)

    ctr_agg.sort_values(by='ctr')

    ctr_agg['dma'] = ctr_agg['dma'].apply(str)

    one_hot = pd.get_dummies(ctr_agg['dma'])#, drop_first=True)
    one_hot = one_hot.astype(object)
    # Drop column B as it is now encoded
    ctr_agg = ctr_agg.drop('dma',axis = 1)
    # Join the encoded df
    ctr_agg = ctr_agg.join(one_hot)

    st.write("""#### 1. lets one hot encode the categorical data
    The categorical variables we're onehot encodeing are 'device_type', 'weekday_user_tz', 'hour_user_tz',
    as when we aggrigate them we'll get a percentage of of occourances in the individual bands.
    """)
    st.write("""#### 2. Aggrigate to the zip plus 1 level
    I chose to aggrigate the values to the zip plus one level as i am aware that there exists US sensus data
    that is take to the zip+1 level meaning that i would be able to to add estimated demographic and 
    finantial feature to the model to evaluate whether they improve the predictivity of the model.

    I also think that just adding all the 100s of features is definatley not an appropriate thing to do
    without potentially knowing more information about the product in this setting
    """)
    st.dataframe(ctr_agg.iloc[:5])

    st.write("""#### 3. 
    Below is an example of the 100s of feature that can be derived from the demorgraphic data to help determine
    why one postcode might be more suspetible to the specific ad. ie they might have a high percentage of single
    males or older women ext.""")


    demographic_data = pd.read_csv(demographic_data_path)
    demographic_data['Estimate!!RACE!!Total population'].fillna(0, inplace=True)
    demographic_data['diversity'] = pd.to_numeric(
            demographic_data['Estimate!!RACE!!Total population!!One race'].replace(
            '-',0), errors='coerce') / demographic_data['Estimate!!RACE!!Total population']


    ctr_agg = pd.merge(
    ctr_agg, demographic_data[
            ['Geographic Area Name', 
            'Estimate!!RACE!!Total population',
            'diversity']
    ], left_on='postal_code', right_on=['Geographic Area Name'], how='left').drop('Geographic Area Name', axis=True)
    ctr_agg.replace([np.inf, -np.inf], 0, inplace=True)

    st.write("""#### 4. 
    Add geographic parameters Below you can see that we added the 2 demographic features to the dataset. These 
    variables being total population and divercity metric calculated by taking the most common race and deviding 
    it by the total population""")

    st.dataframe(ctr_agg.iloc[:5])

    st.write("""#### 5. Sample of correlation matrix
    I had a look to make sure non of the paramers correlated since i'm planning on using logistic regression to 
    initially classify the users. below is a sample of categories in a correlation matrix plot
    """)
    corrMatrix = ctr_agg[['ctr', 'diversity', 'Estimate!!RACE!!Total population']].corr()
    fig = sns.pairplot(corrMatrix)
    st.pyplot(fig)

    for c in ctr_agg.select_dtypes(include=['float', 'int']).columns:
        if c != 'ctr':
            pt = PowerTransformer()
            ctr_agg.loc[:, c] = pt.fit_transform(np.array(ctr_agg[c]).reshape(-1, 1))
    ctr_agg.fillna(0, inplace=True)

    ctr_agg.sort_values(by='ctr')

    X = ctr_agg.drop('ctr', axis=1)
    y = np.ceil(ctr_agg['ctr'])

    model = LogisticRegression(class_weight={0: np.count_nonzero(y==0)/ len(y),
                                            1: np.count_nonzero(y!=0)/ len(y)})
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    grid = dict()
    #grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    grid['l1_ratio'] = np.arange(0, 1, 0.01)
    grid['solver'] = ['saga', 'liblinear']
    search = GridSearchCV(model, grid, scoring='recall', cv=cv, n_jobs=20)
    # perform the search
    results = search.fit(X, y)

    thresh_result = np.array([i[1]for i in results.predict_proba(X)])
    thresh = 0.05
    thresh_result[thresh_result < thresh] = 0
    thresh_result[thresh_result >= thresh] = 1

    st.write("""#### 6. Sample of Confusion matrix
    I've trained a classifier to classify whether a zip+1 will have a ctr value of greater than 0
    or not to see if i can reduce the value of zero in the data set whilst keeping as many instances 
    of ctr as possible. below you can see that a confusion matrix where i've reduced the threshold for
    the prediction of the 1 class. you can see that the accuracy is arround 81% and was closer to 95% 
    before reducing the threshold to increase the recall. Ideally you'd want to have a recall of 100%
    but you have to consider that with a recall of 100% you're likely going to have a much lower precision 
    and will end up with lots of zeros in your results. 
    """)

    fig, ax = plt.subplots() 
    cm = confusion_matrix(y, thresh_result)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    st.pyplot(fig)

    ctr_agg['pred'] = thresh_result

    one_pred = ctr_agg[ctr_agg['pred'] == 1]
    zero_pred = ctr_agg[ctr_agg['pred'] == 0]

    fig, ax = plt.subplots() 
    ax = plt.hist(one_pred['ctr'], bins=50, label='predicted 1', density=True, log=True)
    ax = plt.hist(zero_pred['ctr'], bins=50, label='predicted 0', density=True, log=True)
    plt.xlabel('ctr')
    plt.legend()

    st.write("""#### 7.1 Results from classification
    I've trained a classifier to classify whether a zip+1 will have a ctr value of greater than 0
    or not to see if i can reduce the value of zero in the data set whilst keeping as many instances 
    of ctr as possible. below you can see that a confusion matrix where i've reduced the threshold for
    the prediction of the 1 class. you can see that the accuracy is arround 81% and was closer to 95% 
    before reducing the threshold to increase the recall. Ideally you'd want to have a recall of 100%
    but you have to consider that with a recall of 100% you're likely going to have a much lower precision 
    and will end up with lots of zeros in your results. 
    """)
    st.pyplot(fig)
    st.write("""#### 7.2 Results from classification
    If we have a look at the distribution true negatives and the false negative  we can see that even
    we're capturing the majority of the instances where we have a ctr value of greater 0, we aren't 
    capturing the higher values of ctr. when we decrease the threshold further we start to capture 
    these values but end up making this exercise redundant as we'll inherit a lot more false positves 
    inflating the zeros in our dataset again. 
    """)

    probs_y=results.predict_proba(X) 
    # probs_y is a 2-D array of probability of being labeled as 0 (first column of array) vs 1 (2nd column in array)

    precision, recall, thresholds = precision_recall_curve(y, probs_y[:, 1]) 
    #retrieve probability of being 1(in second column of probs_y)
    pr_auc = metrics.auc(recall, precision)
    fig, ax = plt.subplots() 
    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0,1])
    st.pyplot(fig)

    st.write("""#### 7.2 Results from classification
    If we have a look at the distribution true negatives and the false negative  we can see that even
    we're capturing the majority of the instances where we have a ctr value of greater 0, we aren't 
    capturing the higher values of ctr. when we decrease the threshold further we start to capture 
    these values but end up making this exercise redundant as we'll inherit a lot more false positves 
    inflating the zeros in our dataset again. Below you can the results of increasing decreasing the threshold. 
    """)

    bar_dict = {}
    columns = X.columns.to_list()
    importance = results.best_estimator_.coef_
    for index in range(len(X.columns)):
            bar_dict[columns[index]] = importance[0][index]

    fig, ax = plt.subplots() 
    plt.bar(*zip(*bar_dict.items()))
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(fig)

    st.write("""#### 8. Discussion
    Ultimately I think i could improve the model by adding more features like the ones shown in the 
    employment data below. More over there are more relevent databases that can be found 
    on the US (census](https://www.census.gov/data.html) website. In principle if we are able to remove 
    the obvious zero values with a very high recall we can train a second model to predict the less 
    zero inflated data.""")

    working_data = pd.read_csv(employment_data_path)
    st.dataframe(working_data.iloc[:5])