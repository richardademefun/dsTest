from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dsTest.utils import encode_cat_features, rescale_values
ModelRegressor = Union[LinearRegression, LogisticRegression]

class HurdleModel():

    def __init__(self, clf_name: str = 'logistic', reg_name: str = 'linear', clf_params: Optional[dict] = None, reg_params: Optional[dict] = None):
        """Initialise class"""

        self.clf_name = clf_name
        self.reg_name = reg_name
        self.clf_params = clf_params
        self.reg_params = reg_params

    def run(self, data_path: Union[str, Path], line_item_id: str) -> None:
        """Train the regression model.
        
        Parameters
        ----------
        data_path : Union[str, Path]
            test sample params performance file.
        line_item_id : str
            The line item id you would like to use as a sample for this demo.
        """
        self.preprocess(data_path, line_item_id)
        thresh = self.train_classifier(self.preprocesed_data)
        pred_y = self.trained_classsifier.best_estimator_.predict_proba(self.preprocesed_data.drop(['ctr', 'true_y'], axis=1))[:, 1]
        thresh_result = np.array([i for i in pred_y])
        thresh_result[thresh_result < thresh] = 0
        thresh_result[thresh_result >= thresh] = 1
        self.preprocesed_data['pred_result'] = thresh_result
        self.data_for_regression = self.preprocesed_data[self.preprocesed_data['pred_result']== 1].copy()
        self.data_for_regression.drop(['true_y', 'pred_result'], axis=1, inplace=True)
        self.train_regressor(self.data_for_regression)
        y_pred = self.trained_regressor.predict(self.X_test,exog_infl=self.X_test)

    def train_regressor(self, data: pd.DataFrame) -> None:
        """Train the regression model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to run through the regression model.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(['ctr'], axis=1), 
        data['ctr'], 
        test_size=0.2, 
        random_state=42, 
        stratify=np.ceil(data['ctr']))

        self.trained_regressor = sm.ZeroInflatedPoisson(
            endog=self.y_train, exog=self.X_train, exog_infl=self.X_train, inflation='logit').fit()
        

    def find_classifier_threshold(self, x_vals: pd.DataFrame, y_vals: pd.DataFrame, accuracy: float = 0.001) -> float:
        """Find threshold of test data where true negatives are minimised without losing any true positives
        
        Parameters
        ----------
        x_vals : pd.DataFrame
            Validation X data.
        y_vals : pd.DataFrame
            Validation y data.
        accuracy : float
            step size to find optimal parameter
            
        Returns
        -------
        float
            Threshold of test data where true negatives are minimised without losing any true positives  
        """
        total_occourrences = int(sum(y_vals))
        pred_prbabs = self.trained_classsifier.best_estimator_.predict_proba(x_vals)[:, 1]
        pred_prbabs = np.array([i for i in pred_prbabs])
        thresh = 0.001
        while thresh < 0.5:
            pred_prbabs_copy = pred_prbabs.copy()
            pred_prbabs_copy[pred_prbabs_copy < thresh] = 0
            pred_prbabs_copy[pred_prbabs_copy >= thresh] = 1
            _, _, _, tp = confusion_matrix(y_vals, pred_prbabs_copy).ravel()
            if tp != total_occourrences:
                thresh-=accuracy
                break
            thresh+=accuracy
        return thresh

       
    def train_classifier(self, data: pd.DataFrame, folds: int=3, jobs: int=-1) -> float:
        """Train the classification model.
        
        Parameters
        ----------
        data : pd.DataFrame
            data you would like to train.
        fold : int
            number of of folds in GridSearchCV.
        jobs : int
            number of processors to use.
            
        Returns
        -------
        float
            Threshold of test data where true negatives are minimised without losing any true positives  
        """
        data['true_y'] = np.ceil(data['ctr'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(['true_y', 'ctr'], axis=1), 
        data['true_y'], 
        test_size=0.2, 
        random_state=42, 
        stratify=data['true_y'])
 
        self.classifier = self.initialise_model(self.clf_name)
        search = GridSearchCV(self.classifier, self.clf_params, scoring='recall', cv=folds, n_jobs=jobs)
        search.fit(self.X_train, self.y_train)
        self.trained_classsifier = search
        threshold = self.find_classifier_threshold(self.X_test, self.y_test)
        return threshold
    
    def initialise_model(self, chosen_model: str) -> ModelRegressor:
        """ Choose between multiple models.

        Parameters
        ----------
        chosen_model : str
            Choose one of the modles to train.

        Returns
        -------
        ModelRegressor
            Model that you would like to train 
        """
        models = {'logistic': LogisticRegression()}
        return models[chosen_model]
        
    def preprocess(self, data_path: Union[str, Path], line_item_id: str, demogaphic_data: Union[str, Path]) -> None:
        """Preprocess the input data ready to be used to train models.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data set you would like the demographic data merged with.
        line_item_id : str
            The line item id you would like to use as a sample for this demo      
        """
        categorical_variables_1 = ['device_type', 'weekday_user_tz', 'hour_user_tz', 'city']
        data = pd.read_csv(data_path)
        ctr_sample, _ = self.split_data(data, line_item_id)
        encoded_data = encode_cat_features(ctr_sample, categorical_variables_1)
        agg_data = self.aggrigate_to_zip(encoded_data)
        encoded_data = encode_cat_features(agg_data, ['dma'])
        demographic_enhanced_data = self.add_Demographic_data(encoded_data, demogaphic_data)
        demographic_enhanced_data.replace([np.inf, -np.inf], 0, inplace=True)
        demographic_enhanced_data.dropna(inplace=True, axis=0)
        scaled_data = rescale_values(demographic_enhanced_data)
        scaled_data = scaled_data.loc[:, ~scaled_data.columns.str.contains('^Unnamed')]
        self.preprocesed_data = scaled_data


    def add_Demographic_data(self, ctr_agg: pd.DataFrame, demographic_data_path: Union[Path, str]) -> pd.DataFrame:
        """Add a sample of demographic data to the main dataset.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data set you would like the demographic data merged with.
        demographic_data_path : Union[Path, str]
            Location of the demographic data file.
        Returns
        -------
        pd.DataFrame
            Data set with demographic data.
        """
        demographic_data = pd.read_csv(demographic_data_path, index_col=0)
        demographic_data['Estimate!!RACE!!Total population'].fillna(0, inplace=True)
        demographic_data['diversity'] = pd.to_numeric(
            demographic_data['Estimate!!RACE!!Total population!!One race'].replace(
                '-', 0), errors='coerce') / demographic_data['Estimate!!RACE!!Total population']
        ctr_agg = pd.merge(
        ctr_agg, demographic_data[
            ['Geographic Area Name',
             'Estimate!!RACE!!Total population',
             'diversity']
        ], left_on='postal_code', right_on=['Geographic Area Name'], how='left').drop('Geographic Area Name', axis=1)
        return ctr_agg
    
    def aggrigate_to_zip(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply custom aggrigation to teh dataset
        
        Parameters
        ----------
        data : pd.DataFrame
            The input test parameter data set.

        Returns
        -------
        pd.DataFrame
            Data aggrigate to the zip+1 level.
        """
        ctr_count = data.groupby(
            "postal_code").agg(
                {'postal_code': 'count',
                'dma': 'first'})

        ctr_agg = data.groupby(
            "postal_code").agg(
            'sum')
        ctr_agg['dma'] = ctr_count['dma']
        ctr_agg['count'] = ctr_count['postal_code']
        for col in ctr_agg.columns:
            if col != 'dma':
                ctr_agg[col] = ctr_agg[col] / ctr_agg['count']
        ctr_agg.drop('count', inplace=True, axis=1)
        ctr_agg['dma'] = ctr_agg['dma'].apply(str)
        return ctr_agg

    def split_data(self, data: pd.DataFrame, line_item_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into it's ctr and cpa sections respectively.
        
        Parameters
        ----------
        data : pd.DataFrame
            The input test parameter data set.
        line_item_id : str
            The line item id you would like to use as a sample for this demo

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The split ctr and cpa data sets.
        """
        sample_data = data[data['line_item_id'] == line_item_id]
        ctr_data = sample_data[sample_data['pixel_id'] == 0].drop(
            ['advertiser_id', 'insertion_order_id', 'line_item_id', 'region','pixel_id',
            'impressions', 'clicks', 'booked_revenue_adv_curr', 'booked_revenue', 'conversions',
            'rpm', 'rpa_adv_curr', 'country'], axis=1)

        cpa_data = sample_data[~sample_data['pixel_id'] == 0]# filter cpa data later
        return ctr_data, cpa_data   

if __name__ == "__main__":
    grid = dict()
    # grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    grid['l1_ratio'] = np.arange(0, 1, 0.01)
    grid['solver'] = ['saga', 'liblinear']
    test = HurdleModel(clf_params=grid)


