'''
    Final Project
    Joseph Nelson Farrell & Michael Missone
    DS 5230 Unsupervised Machine Learning
    Northeastern University
    Steven Morin, PhD

    This file contains a function that will return a list of column names whose proportion of missing attributes is greater than a specified
    threshold.
'''
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def missingness_cols(df: pd.DataFrame, threshdold: float) -> list:
    '''
        Function: missingness_cols
        Parameters: 1 pd.Dataframe, 1 float
            df: the dataframe whose columns will checked for missingess
            threshold: the threshold proportion to determine if column should be dropped
        Returns: 1 list
            cols_to_drop: the columns whose missingness exceeds threshold. 

        The function will find the proportion of missingness for each column in a dataframe and return a list of 
        columns whose missingness proportion exceeds the threshold
    '''
    # instantiate drop list
    cols_to_drop = []

    # iterate over columns
    for col in df.columns:

        # find missingness proportion
        missing_proportion = df[col].isna().sum() / df.shape[0]

        # check against threshold; update list
        if missing_proportion >= threshdold:
            cols_to_drop.append(col)

    return cols_to_drop

def display_missingness(df: pd.DataFrame, save_path:str = None, save_fig:bool = False) -> None:
    """
    Displays the missingness of a dataframe 
    """
    msno.matrix(df)
    plt.title("Data Missingness", fontsize = 40, weight = "bold", y = 1.03)
    plt.ylabel("Hospital Respondents", fontsize = 20, weight = "bold")
    plt.xlabel("Survey Questions", fontsize = 20, weight = "bold")
    if save_fig:
        plt.savefig(save_path);

if __name__ == "__main__":
    pass