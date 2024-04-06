import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def to_int(df):
    """Helper for pipeline conversion to int

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: df with converted columns
    """
    df['BedroomAbvGr'] = df['BedroomAbvGr'].astype(int)
    return df


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


def process_data(df):
    """ binning of year features.
    """
    # drop columns with too many nans
    df.drop('EnclosedPorch', axis=1, inplace=True)
    df.drop('WoodDeckSF', axis=1, inplace=True)

    # apply binning to relevant columns
    df['GarageYrBlt'] = df['GarageYrBlt'].apply(binning)
    df['YearBuilt'] = df['YearBuilt'].apply(binning)
    df['YearRemodAdd'] = df['YearRemodAdd'].apply(binning)
    return df


def plotRegResult(model, x_train, y_train, x_test, y_test, r2_test=None, r2_train=None):
    """Plot train test model performance in two subplots"""
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    sns.scatterplot(ax=ax[0],
                    x=y_test,
                    y=model.predict(x_test),
                    )
    ax[0].plot(range(700000), range(700000), 'k')
    try:
        ax[0].title.set_text(f'Test: r2={r2_test:0.2f}')
    except:
        ax[0].title.set_text(f'Test')
    ax[0].set_ylabel('predicted SalePrice')

    sns.scatterplot(ax=ax[1], x=y_train, y=model.predict(x_train))
    try:
        ax[1].title.set_text(f'Train: r2={r2_train:0.2f}')
    except:
        ax[1].title.set_text(f'Train')
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
