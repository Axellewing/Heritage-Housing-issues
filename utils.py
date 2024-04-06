import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


def binning(year):
    """ Binning categorical variables with year as an
    independent variable."""
    if pd.notna(year):
        year = int(year)
        if 1870 < year <= 1880:
            return "1871-1880"
        elif 1880 < year <= 1890:
            return "1881-1890"
        elif 1890 < year <= 1900:
            return "1891-1900"
        elif 1900 < year <= 1910:
            return "1901-1910"
        elif 1910 < year <= 1920:
            return "1911-1920"
        elif 1920 < year <= 1930:
            return "1921-1930"
        elif 1930 < year <= 1940:
            return "1931-1940"
        elif 1940 < year <= 1950:
            return "1941-1950"
        elif 1950 < year <= 1960:
            return "1951-1960"
        elif 1960 < year <= 1970:
            return "1961-1970"
        elif 1970 < year <= 1980:
            return "1971-1980"
        elif 1980 < year <= 1990:
            return "1981-1990"
        elif 1990 < year <= 2000:
            return "1991-2000"
        elif 2000 < year <= 2010:
            return "2001-2010"
        else:
            return "Unknown"
    else:
        return "Unknown"


# convert integer back to np.nan
def back_to_nan(row, n):
    if row == n:
        return np.nan
    return int(row)


def plotRegResult(df_test, y_train, y_train_pred, r2_test=None, r2_train=None):
    """Plot train test model performance in two subplots"""
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    sns.scatterplot(ax=ax[0],
                    x=df_test['SalePrice'],
                    y=df_test['y_pred'],
                    )
    ax[0].plot(range(700000), range(700000), 'k')
    ax[0].title.set_text(f'Test: r2={r2_test:0.2f}')
    ax[0].set_ylabel('predicted SalePrice')

    sns.scatterplot(ax=ax[1], x=y_train, y=y_train_pred)
    ax[1].title.set_text(f'Train: r2={r2_train:0.2f}')
    ax[1].plot(range(700000), range(700000), 'k')
    ax[1].set_ylabel('predicted SalePrice')
    plt.tight_layout()
    return fig


def increase_feature(df, col_name: str):
    """ Increase numerical variable by 10% or categorical variable
    by one point.
    """
    if df[col_name].dtype in [np.dtype(np.int32), np.dtype(np.int64)]:  # categorical
        df[col_name] = df[col_name] + 1
    elif df[col_name].dtype in [np.dtype(np.float32), np.dtype(np.float64)]:
        df[col_name] = df[col_name] * 1.1  # 10% increase
    else:
        pass

    return df


def increase_vs_base(base_pred, new_pred):
    diff = new_pred - base_pred
    rel_increase = diff / base_pred * 100  # in percents
    return diff, rel_increase

    
def runRegression(DF, test_size=0.2, n_est=500, lr=0.01):
    """Run whole regression from the dataframe, test-train split,
    k-features selector, fit linear regression, make train test predictions,
    calculate r2 score.

    Args:
        DF: dataframe
        test_size (float): ratio of test samples for split
        k (int): If int, defines how many features are selected from the DF. Defaults to None.
    Returns:
        df_test (pd.DataFrame): test split part of DF
        y_train: training values, train
        y_train_pred: model predictions from train data
        r2_test: r2 score on test data
        r2_train: r2 score on train data
        X_train, X_test, y_test: Rest of the output from train_test_split()
    """
    X = DF.drop("SalePrice", axis=1)
    y = DF["SalePrice"]
    # split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size,
                                                        random_state=42,
                                                        )
        
    xgb = XGBRegressor(n_estimators=n_est, learning_rate=lr)
    xgb.fit(X_train, y_train)
    y_test_pred = xgb.predict(X_test)

    # predict from model, input is only X_test, y_pred should be close to y_test, wchich is the round truth
    y_test_pred = xgb.predict(X_test)
    # making it into a series with correct index
    sy_pred = pd.Series(y_test_pred, index=X_test.index, name='y_pred')
    
    # merge predictions on test DF
    df_test = DF.merge(sy_pred, left_index=True, right_index=True)
    df_test = df_test.merge(X_test)
    r2_test = r2_score(y_true=df_test['SalePrice'],
                       y_pred=df_test['y_pred'],
                       sample_weight=None)
    
    # Train predictions
    # predict from model, input is only X_test, y_pred should be close to y_test, wchich is the ground truth
    y_train_pred = xgb.predict(X_train)
    r2_train = r2_score(y_true=y_train,
                        y_pred=y_train_pred,
                        sample_weight=None)
    return df_test, y_train, y_train_pred, r2_test, r2_train, X_train, X_test, y_train, y_test