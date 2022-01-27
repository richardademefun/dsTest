from typing import List, Union
import pandas as pd 
from sklearn import preprocessing
import joblib
from pathlib import Path

def rescale_values(self, data: pd.DataFrame, output: Union[str, Path], save: bool = False) -> pd.DataFrame:
    """Rescale the continuous data
    
    Parameters
    ----------
    data : pd.DataFrame
        The entire dataset including continuous variables 
    save : List[str]
        If true save scaler(you'll probably need it later).

    Returns
    -------
    pd.DataFrame
        The initial dataset with the specified categorical columns on hot encoded
    """
    col_to_scale = data.select_dtypes(include=['float', 'int']).columns.tolist()
    col_to_scale.remove('ctr')
    scaler = preprocessing.MinMaxScaler()
    data[col_to_scale] = scaler.fit_transform(data[col_to_scale])
    if save:
        joblib.dump(scaler, output)
    data.fillna(0, inplace=True)
    return data


def encode_cat_features(self, data: pd.DataFrame, categorical_variables: List[str]) -> pd.DataFrame:
        """One hot encode categorical variables.
        
        Parameters
        ----------
        data : pd.DataFrame
            The entire dataset.
        categorical_variables : List[str]
            The columns the within the data that you would like to one hot encode.

        Returns
        -------
        pd.DataFrame
            The initial dataset with the specified categorical columns on hot encoded
        """
        data[categorical_variables] = data[categorical_variables].astype(object)
        one_hot = pd.get_dummies(data[categorical_variables], drop_first=True)
        one_hot = one_hot.astype(object)
        data.drop(categorical_variables, axis=1, inplace=True)
        return data.join(one_hot)

