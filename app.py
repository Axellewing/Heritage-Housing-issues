import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import load
from utils import (increase_feature, increase_vs_base,
                   plotRegResult, to_int, process_data)
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)


### loading preprocessing and model pipeline
model = load('model.save')

numerics = ['1stFlrSF', 'GarageArea', 'GrLivArea', 'TotalBsmtSF']
categories = ['BedroomAbvGr', 'BsmtExposure', 'BsmtFinType1',
              'GarageFinish', 'GarageYrBlt', 'KitchenQual',
              'OverallCond', 'OverallQual', 'YearBuilt', 'YearRemodAdd']
target=['SalePrice']
columns = numerics + categories

house_dict = {'House 1': 0,
              'House 2': 1,
              'House 3': 2,
              'House 4': 3,}

### LOADING DFs ###
df = pd.read_csv("house_prices_records.csv")
df = process_data(df)
df = df[columns+target]


# processing inherited properties.
df_inh = pd.read_csv('inherited_houses.csv')
df_base = df_inh.copy()
df_base = process_data(df_base)
df_base = df_base[columns]

# baseline prediction, 
base_pred = model.predict(df_base)

def show_house_potential(house):
# increase feature one by one and evaluate predicted increase in price
    diff = []
    max_inc = -1e6
    for column in columns:
        df2 = df_inh.copy()
        df2 = increase_feature(df2, column)
        df2 = process_data(df2)
        df2 = df2[columns]
        pred = model.predict(df2)

        abs_diff, rel_increase = increase_vs_base(base_pred, pred)
        idx = house_dict[house]
        diff.append(abs_diff[idx])
        if rel_increase[idx] > max_inc:
            max_inc = rel_increase[idx]
            golden_col = column
    st.write(f'Current price estimate is **{base_pred[idx]:.2f} $**.')
    
    st.write(f'Max relative increase of **{max_inc:.2f} %** on sale price achieved\
             if **{golden_col}** is improved.')
    
    # plot of absolute increases in property price.
    fig, ax = plt.subplots(figsize=(8, 6))
    print(diff, columns)
    sns.barplot(x=columns, y=diff)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('gain in $')
    ax.set_xlabel('categories')
    ax.set_title('Gains upon house refurbishing')
    plt.tight_layout()
    st.pyplot(fig)




#################
### Dashboard ###
#################
st.title("Heritage Housing Issues Dashboard")
page = st.sidebar.selectbox("Select a Page", ["Overview", "Exploratory Analysis", "Predictive Model Insights",
                                              'Predict your house price'])


# --- Overview Page ---
if page == "Overview":
    st.header("Overview")
    st.write('---')
    st.write("Welcome to the Heritage Housing Issues dashboard! This dashboard aims to provide insights and predictions related to house prices in Ames, Iowa, USA.")

    st.header("Client")
    st.markdown("""
    - Lydia has inherited four houses in Ames, Iowa, USA, presenting an opportunity for financial gain or loss.
    - Lydia, having expertise in Belgian property prices, faces uncertainty about the factors influencing house prices in Ames, Iowa.
    - Lydia has found a public dataset with house prices for Ames, Iowa, providing a basis for analysis and prediction.
    """)

    st.header("Business Case")
    st.markdown("""
    - Understand how house attributes correlate with sale prices.
    - Predict the sales price for inherited houses and any other house in Ames.
    - Show how increase of feature values (10\% for continuous features) and one step increase in categorically features translate into the sale price increase.
                ***Hypothesis: Only small amount of features are worth improving on the inhereted property. With an objective to increase the house sell price
                with a renovation, we want to provide insight into price increase upon improved specifications of the inhereted property.***
    """)

    st.header("Data Science Objectives")
    st.subheader("Functional Requirements")
    st.markdown("""
    - Implement data analysis and visualization.
    - Build a predictive model for sales prices using machine learning techniques.
    - Deploy a Streamlit dashboard to present the findings and predictions.
    """)

    st.subheader("Non-Functional Requirements")
    st.markdown("""
    - Ensure data quality and relevance for accurate predictions.
    - Use version control (Git & GitHub) for effective collaboration and tracking changes.
    - Implement clean and robust code following best practices.
    """)

    st.subheader("Success Criteria")
    st.markdown("""
    - Achieve accurate predictions of sales prices with the goal of obtaining a high R2 score.
    - Create an informative and user-friendly Streamlit dashboard.
    - Successfully deploy the dashboard on Heroku.
    """)


# --- Correlation Analysis Page ---    
if page == "Exploratory Analysis":
    st.header("Exploratory Analysis")
    st.write('---')

    st.subheader("Pairwise Scatter Plots for Numerical Variables")
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    for i, col in enumerate(numerics):
        sns.scatterplot(ax=ax[i//2, i%2], data=df, x=col, y='SalePrice')
        ax[i//2, i%2].set_xlabel(col)
        ax[i//2, i%2].set_ylabel('SalePrice')
        ax[i//2, i%2].set_title(f"SalePrice vs {col}")
    plt.tight_layout()
    st.pyplot()

    
    st.subheader("Correlation Heatmap for Numerical Variables")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerics+target].corr()[target], annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlations Between Variables", size=15)
    plt.tight_layout()
    st.pyplot()

    st.subheader("Conclusion")
    st.write("The correlation of these numerical variables is greater than 0.5, indicating a significant dependency of the sale price on these features.")

    st.title("SalePrice vs Categorical Variables")
    fig, ax = plt.subplots(5, 2, figsize=(10, 16), sharey=True)
    for i, category in enumerate(categories):
        sns.boxplot(ax=ax[i//2, i%2], x=category, y=target[0], data=df, palette="Set2")
        ax[i//2, i%2].set_title(f'{target[0]} vs {category} categories')
        ax[i//2, i%2].set_xlabel(category)
        ax[i//2, i%2].set_ylabel(target[0])
        ax[i//2, i%2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Conclusion")
    st.write("""
            - We can observe the variation of means of 'SalePrice' from one category to another for most features.
             Categorical variables are correlated with the target.
            - Not surprisingly, the overall quality (OverallQual) assessment present the clearest correlation with the sale price.
            """)

# --- Predictive Model Page ---    
if page == "Predictive Model Insights":
    st.header("Predictive Model Insights")
    st.write('---')
    
    st.write('Based on the model, the following is expected upon increase of 10 percent\
             of the cont. variables or 1 point in the categorical ones.')
    
    house = st.radio(
        "Select inherited house for analysis",
        key="house_selector",
        options=["House 1", "House 2", "House 3", "House 4"],
    )
    
    show_house_potential(house)

    st.subheader("Conclusion")
    st.write("""
            - Hypothesis has been quite well confirmed. It seems that investment on overall
            quality of the property might yield a good return value on the renovation
             investment strategy.
            - Beware however, that the features are not independent,
            area of first floor for example increases the total area too,
            and so on.
            """)

    st.subheader("Overall model performance")
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    fig_model = plotRegResult(model, X_train, y_train, X_test, y_test)
    st.pyplot(fig_model)

    st.write("""
            - Model performance is decent, with almost 85% variance explained for the test data.
             The model is also robust as shown in the notebook.
            """)

# --- Predict your house price ---    
if page == "Predict your house price":
    st.header("Predict your house price")
    st.write('---')

    _1stFlrSF = st.slider("Select  the area of the first floor in square feet (1stFlrSF)", 
                          min_value=334.0, max_value=4692.0, value=334.0, step=0.01)
    st.write(f"Selected 1stFlrSF Value: {_1stFlrSF}")
    st.write('---')

    GarageArea = st.slider("Select  the size of the garage in square feet (GarageArea)", 
                          min_value=0.0, max_value=1418.0, value=100.0, step=0.01)
    st.write(f"Selected GarageArea Value: {GarageArea}")
    st.write('---')

    GrLivArea = st.slider("Select  the above-grade living area in square feet (GrLivArea)", 
                          min_value=334.0, max_value=5642.0, value=334.0, step=0.01)
    st.write(f"Selected GrLivArea Value: {GrLivArea}")
    st.write('---')

    TotalBsmtSF = st.slider("Select  the total square feet of the basement area (TotalBsmtSF)", 
                          min_value=0.0, max_value=6110.0, value=100.0, step=0.01)
    st.write(f"Selected TotalBsmtSF Value: {TotalBsmtSF}")
    st.write('---')

    BedroomAbvGr = st.slider("Select the number of bedrooms above ground, excluding those in the basement", 
                             min_value=0, max_value=8, value=0, step=1)
    st.write(f"Selected BedroomAbvGr Value: {BedroomAbvGr}")
    st.write('---')

    BsmtExposure = st.slider("Select the level of exposure of the basement walls", 
                             min_value=0, max_value=4, value=0, step=1)
    if BsmtExposure == 0:
        st.write(f"Selected BsmtExposure Value: Av (Average Exposure)")
        _bsmtex = 'Av'
    elif BsmtExposure == 1:
        st.write(f"Selected BsmtExposure Value: Gd (Good Exposure)")
        _bsmtex = 'Gd'
    elif BsmtExposure == 2:
        st.write(f"Selected BsmtExposure Value: Mn (Minimum Exposure)")
        _bsmtex = 'Mn'
    elif BsmtExposure == 3:
        st.write(f"Selected BsmtExposure Value: No (No Exposure)")
        _bsmtex = 'No'
    else:
        st.write(f"Selected BsmtExposure Value: None (No Basement)")
        _bsmtex = 'None'

    st.write('---')

    BsmtFinType1 = st.slider("Select the quality rate of the finished area in the basement", 
                             min_value=0, max_value=6, value=0, step=1)
    if BsmtFinType1 == 0:
        st.write(f"Selected BsmtFinType1 Value: ALQ (Average Living Quarters)")
        _bsmtFi = 'ALQ'
    elif BsmtFinType1 == 1:
        st.write(f"Selected BsmtFinType1 Value: BLQ (Below Average Living Quarters)")
        _bsmtFi = 'BLQ'
    elif BsmtFinType1 == 2:
        st.write(f"Selected BsmtFinType1 Value: GLQ (Good Living Quarters)")
        _bsmtFi = 'GLQ'
    elif BsmtFinType1 == 3:
        st.write(f"Selected BsmtFinType1 Value: LwQ (Low Quality)")
        _bsmtFi = 'LwQ'
    elif BsmtFinType1 == 4:
        st.write(f"Selected BsmtFinType1 Value: None (No Basement)")
        _bsmtFi = 'None'
    elif BsmtFinType1 == 5:
        st.write(f"Selected BsmtFinType1 Value: Rec (Average Rec Room)")
        _bsmtFi = 'Rec'
    else:
        st.write(f"Selected BsmtFinType1 Value: Unf (Unfinished)")
        _bsmtFi = 'Unf'

    st.write('---')

    GarageFinish = st.select_slider("Select the interior finish of the garage", 
                             options=('Rough', 'Unfinished', 'Finished', 'None'), value='None')
    if GarageFinish == 'Finished':
        st.write(f"Selected GarageFinish Value: Finished")
        _garageFinish = 'Fin'
    elif GarageFinish == 'Rough':
        st.write(f"Selected GarageFinish Value: Rough")
        _garageFinish = 'RFn'
    elif GarageFinish == 'Unfinished':
        st.write(f"Selected GarageFinish Value: Unfinished")
        _garageFinish = 'Unf'
    else:
        st.write(f"Selected GarageFinish Value: None")
        _garageFinish = 'None'

    st.write('---')

    GarageYrBlt = st.slider("Select the decade the garage was built", 
                            min_value=0, max_value=12, value=0, step=1)
    if GarageYrBlt == 0 :
        st.write(f"Selected GarageYrBlt Value: 1891-1900")
        _gyear = '1891-1900'
    elif GarageYrBlt == 1 :
        st.write(f"Selected GarageYrBlt Value: 1901-1910")
        _gyear = '1901-1910'
    elif GarageYrBlt == 2 :
        st.write(f"Selected GarageYrBlt Value: 1911-1920")
        _gyear = '1911-1920'
    elif GarageYrBlt == 3 :
        st.write(f"Selected GarageYrBlt Value: 1921-1930")
        _gyear = '1921-1930'
    elif GarageYrBlt == 4 :
        st.write(f"Selected GarageYrBlt Value: 1931-1940")
        _gyear = '1931-1940'
    elif GarageYrBlt == 5 :
        st.write(f"Selected GarageYrBlt Value: 1941-1950")
        _gyear = '1941-1950'
    elif GarageYrBlt == 6 :
        st.write(f"Selected GarageYrBlt Value: 1951-1960")
        _gyear = '1951-1960'
    elif GarageYrBlt == 7 :
        st.write(f"Selected GarageYrBlt Value: 1961-1970")
        _gyear = '1961-1970'
    elif GarageYrBlt == 8 :
        st.write(f"Selected GarageYrBlt Value: 1971-1980")
        _gyear = '1971-1980'
    elif GarageYrBlt == 9 :
        st.write(f"Selected GarageYrBlt Value: 1981-1990")
        _gyear = '1981-1990'
    elif GarageYrBlt == 10 :
        st.write(f"Selected GarageYrBlt Value: 1991-2000")
        _gyear = '1991-2000'
    elif GarageYrBlt == 11 :
        st.write(f"Selected GarageYrBlt Value: 2001-2010")
        _gyear = '2001-2010'
    else :
        st.write(f"Selected GarageYrBlt Value: Unkonwn")
        _gyear = 'Unknown'
    
    st.write('---')

    KitchenQual = st.select_slider("Select the quality rate of the kitchen", 
                            options=('Excellent', 'Good', 'Fair', 'Average'), value='Average')
    if KitchenQual == 'Excellent':
        st.write(f"Selected KitchenQual Value: Excellent")
        _kitchenQ = 'Ex'
    elif KitchenQual == 'Fair':
        st.write(f"Selected KitchenQual Value: Fair")
        _kitchenQ = 'Fa'
    elif KitchenQual == 'Good':
        st.write(f"Selected KitchenQual Value: Good")
        _kitchenQ = 'Gd'
    else:
        st.write(f"Selected KitchenQual Value: Typical/Average")
        _kitchenQ = 'TA'

    st.write('---')

    OverallCond  = st.slider("Select the overall condition rate of the house on a scale from 1 (Very Poor) to 10 (Very Excellent)", 
                             min_value=1, max_value=10, value=1, step=1)
    st.write(f"Selected OverallCond Value: {OverallCond}")
    st.write('---')

    OverallQual  = st.slider("Select the overall material and finish rates of the house on a scale from 1 (Very Poor) to 10 (Very Excellent)", 
                             min_value=1, max_value=10, value=1, step=1)
    st.write(f"Selected OverallQual Value: {OverallQual}")
    st.write('---')

    YearBuilt = st.slider("Select the original construction decade of the house", 
                          min_value=0, max_value=13, value=1, step=1)
    if YearBuilt == 0 :
        st.write(f"Selected YearBuilt Value: 1871-1880")
        _ybuilt = '1871-1880'
    elif YearBuilt == 1 :
        st.write(f"Selected YearBuilt Value: 1881-1890")
        _ybuilt = '1881-1890'
    elif YearBuilt == 2 :
        st.write(f"Selected YearBuilt Value: 1891-1900")
        _ybuilt = '1891-1900'
    elif YearBuilt == 3 :
        st.write(f"Selected YearBuilt Value: 1901-1910")
        _ybuilt = '1901-1910'
    elif YearBuilt == 4 :
        st.write(f"Selected YearBuilt Value: 1911-1920")
        _ybuilt = '1911-1920'
    elif YearBuilt == 5 :
        st.write(f"Selected YearBuilt Value: 1921-1930")
        _ybuilt = '1921-1930'
    elif YearBuilt == 6 :
        st.write(f"Selected YearBuilt Value: 1931-1940")
        _ybuilt = '1931-1940'
    elif YearBuilt == 7 :
        st.write(f"Selected YearBuilt Value: 1941-1950")
        _ybuilt = '1941-1950'
    elif YearBuilt == 8 :
        st.write(f"Selected YearBuilt Value: 1951-1960")
        _ybuilt = '1951-1960'
    elif YearBuilt == 9 :
        st.write(f"Selected YearBuilt Value: 1961-1970")
        _ybuilt = '1961-1970'
    elif YearBuilt == 10 :
        st.write(f"Selected YearBuilt Value: 1971-1980")
        _ybuilt = '1971-1980'
    elif YearBuilt == 11 :
        st.write(f"Selected YearBuilt Value: 1981-1990")
        _ybuilt = '1981-1990'
    elif YearBuilt == 11 :
        st.write(f"Selected YearBuilt Value: 1991-2000")
        _ybuilt = '1991-2000'
    else :
        st.write(f"Selected YearBuilt Value: 2001-2010")
        _ybuilt = '2001-2010'

    st.write('---')

    YearRemodAdd = st.slider("Select the decade of remodeling or additions, with the same value as the construction date if no remodeling occurred", 
                             min_value=0, max_value=6, value=0, step=1)
    if YearRemodAdd == 0 :
        st.write(f"Selected YearRemodAdd Value: 1941-1950")
        _yremod = '1941-1950'
    elif YearRemodAdd == 1 :
        st.write(f"Selected YearRemodAdd Value: 1951-1960")
        _yremod = '1951-1960'
    elif YearRemodAdd == 2 :
        st.write(f"Selected YearRemodAdd Value: 1961-1970")
        _yremod = '1961-1970'
    elif YearRemodAdd == 3 :
        st.write(f"Selected YearRemodAdd Value: 1971-1980")
        _yremod = '1971-1980'
    elif YearRemodAdd == 4 :
        st.write(f"Selected YearRemodAdd Value: 1981-1990")
        _yremod = '1981-1990'
    elif YearRemodAdd == 5 :
        st.write(f"Selected YearRemodAdd Value: 1991-2000")
        _yremod = '1991-2000'
    else :
        st.write(f"Selected YearRemodAdd Value: 2001-2010")
        _yremod = '2001-2010'

    st.write('---')

    if st.button("Predict Sale Price"):
        # values from slider, user input 
        user_values = [_1stFlrSF, GarageArea, GrLivArea, TotalBsmtSF, BedroomAbvGr, _bsmtex, _bsmtFi,
                       _garageFinish, _gyear, _kitchenQ, OverallCond, OverallQual, _ybuilt, _yremod]
        user_input_df = pd.DataFrame([user_values], columns=columns)  # convert to DF

        # predict price
        prediction = model.predict(user_input_df)

        
        # Predicted price
        st.success(f"Predicted Sale Price: ${prediction[0]:,.0f}.")
