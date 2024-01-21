import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Heritage Housing Issues Dashboard")
page = st.sidebar.selectbox("Select a Page", ["Overview", "Correlation Analysis", "Predictive Model Insights"])

# --- Overview Page ---
if page == "Overview":
    st.header("Overview")
    st.write("Welcome to the Heritage Housing Issues dashboard! This dashboard aims to provide insights and predictions related to house prices in Ames, Iowa, USA.")

    st.header("Project Summary")
    st.markdown("A fictional individual, Lydia Doe, has inherited four houses in Ames, Iowa, USA. The goal is to predict the sales prices for these inherited houses and any other house in Ames. The project involves data analysis, visualization, and the development of a predictive model.")

    st.header("Business Objectives")
    st.markdown("""
    - Understand how house attributes correlate with sale prices.
    - Predict the sales price for inherited houses and any other house in Ames.
    """)

    st.header("Situation Assessment")
    st.markdown("""
    Lydia, originally from Belgium, is uncertain about the local housing market in Ames, Iowa, and seeks assistance to maximize the sales prices of her inherited houses. The dataset, sourced from Kaggle, contains information about house prices and various attributes in Ames.
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
if page == "Correlation Analysis":
    st.header("Correlation Analysis")

    df = pd.read_csv("for_analysis.csv")
    numerics = ['1stFlrSF', 'GarageArea', 'GrLivArea', 'TotalBsmtSF']
    categories = ['BedroomAbvGr', 'BsmtExposure', 'BsmtFinType1', 'GarageFinish', 'GarageYrBlt', 'KitchenQual', 'OverallCond', 'OverallQual',
                  'YearBuilt', 'YearRemodAdd']
    target=['SalePrice']

    st.subheader("Pairwise Scatter Plots for Numerical Variables")
    for col in numerics:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=col, y='SalePrice')
        plt.xlabel(col)
        plt.ylabel('SalePrice')
        plt.title(f"Scatter Plot of SalePrice with {col}")
        st.pyplot()

    
    st.subheader("Correlation Heatmap for Numerical Variables")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerics+target].corr()[target], annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlations Between Variables", size=15)
    st.pyplot()

    st.subheader("Conclusion")
    st.write("The correlation of these numerical variables is greater than 0.5, indicating a significant dependency of the sale price on these features.")

    st.title("Distribution of SalePrice by Categorical Variables")
    for category in categories:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=category, y=target[0], data=df, palette="Set2", ax=ax)
        plt.title(f'Distribution of {target[0]} by {category}')
        plt.xlabel(category)
        plt.ylabel(target[0])
        st.pyplot(fig)

    st.subheader("Conclusion")
    st.write("The median of each category of these categorical variables is changing, indicating a significant dependency of the sale price on these features.")
    

# --- Predictive Model Page ---    
if page == "Predictive Model Insights":
    st.header("Predictive Model Insights")
    st.write('---')

    _1stFlrSF = st.slider("Select  the area of the first floor in square feet (1stFlrSF)", 
                          min_value=334.0, max_value=4692.0, value=334.0, step=0.01)
    st.write(f"Selected 1stFlrSF Value: {_1stFlrSF}")
    st.write('---')

    GarageArea = st.slider("Select  the size of the garage in square feet (GarageArea)", 
                          min_value=0.0, max_value=1418.0, value=0.0, step=0.01)
    st.write(f"Selected GarageArea Value: {GarageArea}")
    st.write('---')

    GrLivArea = st.slider("Select  the above-grade living area in square feet (GrLivArea)", 
                          min_value=334.0, max_value=5642.0, value=334.0, step=0.01)
    st.write(f"Selected GrLivArea Value: {GrLivArea}")
    st.write('---')

    TotalBsmtSF = st.slider("Select  the total square feet of the basement area (TotalBsmtSF)", 
                          min_value=0.0, max_value=6110.0, value=0.0, step=0.01)
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
    elif BsmtExposure == 1:
        st.write(f"Selected BsmtExposure Value: Gd (Good Exposure)")
    elif BsmtExposure == 2:
        st.write(f"Selected BsmtExposure Value: Mn (Minimum Exposure)")
    elif BsmtExposure == 3:
        st.write(f"Selected BsmtExposure Value: No (No Exposure)")
    else:
        st.write(f"Selected BsmtExposure Value: None (No Basement)")

    st.write('---')

    BsmtFinType1 = st.slider("Select the quality rate of the finished area in the basement", 
                             min_value=0, max_value=6, value=0, step=1)
    if BsmtFinType1 == 0:
        st.write(f"Selected BsmtFinType1 Value: ALQ (Average Living Quarters)")
    elif BsmtFinType1 == 1:
        st.write(f"Selected BsmtFinType1 Value: BLQ (Below Average Living Quarters)")
    elif BsmtFinType1 == 2:
        st.write(f"Selected BsmtFinType1 Value: GLQ (Good Living Quarters)")
    elif BsmtFinType1 == 3:
        st.write(f"Selected BsmtFinType1 Value: LwQ (Low Quality)")
    elif BsmtFinType1 == 4:
        st.write(f"Selected BsmtFinType1 Value: None (No Basement)")
    elif BsmtFinType1 == 5:
        st.write(f"Selected BsmtFinType1 Value: Rec (Average Rec Room)")
    else:
        st.write(f"Selected BsmtFinType1 Value: Unf (Unfinished)")

    st.write('---')

    GarageFinish = st.slider("Select the interior finish of the garage", 
                             min_value=0, max_value=3, value=0, step=1)
    if GarageFinish == 0:
        st.write(f"Selected GarageFinish Value: Fin (Finished)")
    elif GarageFinish == 1:
        st.write(f"Selected GarageFinish Value: None (No Garage)")
    elif GarageFinish == 2:
        st.write(f"Selected GarageFinish Value: RFn (Rough Finished)")
    else:
        st.write(f"Selected GarageFinish Value: Unf (Unfinished)")

    st.write('---')

    GarageYrBlt = st.slider("Select the decade the garage was built", 
                            min_value=0, max_value=12, value=0, step=1)
    if GarageYrBlt == 0 :
        st.write(f"Selected GarageYrBlt Value: 1891-1900")
    elif GarageYrBlt == 1 :
        st.write(f"Selected GarageYrBlt Value: 1901-1910")
    elif GarageYrBlt == 2 :
        st.write(f"Selected GarageYrBlt Value: 1911-1920")
    elif GarageYrBlt == 3 :
        st.write(f"Selected GarageYrBlt Value: 1921-1930")
    elif GarageYrBlt == 4 :
        st.write(f"Selected GarageYrBlt Value: 1931-1940")
    elif GarageYrBlt == 5 :
        st.write(f"Selected GarageYrBlt Value: 1941-1950")
    elif GarageYrBlt == 6 :
        st.write(f"Selected GarageYrBlt Value: 1951-1960")
    elif GarageYrBlt == 7 :
        st.write(f"Selected GarageYrBlt Value: 1961-1970")
    elif GarageYrBlt == 8 :
        st.write(f"Selected GarageYrBlt Value: 1971-1980")
    elif GarageYrBlt == 9 :
        st.write(f"Selected GarageYrBlt Value: 1981-1990")
    elif GarageYrBlt == 10 :
        st.write(f"Selected GarageYrBlt Value: 1991-2000")
    elif GarageYrBlt == 11 :
        st.write(f"Selected GarageYrBlt Value: 2001-2010")
    else :
        st.write(f"Selected GarageYrBlt Value: Unkonwn")
    
    st.write('---')

    KitchenQual = st.slider("Select the quality rate of the kitchen", 
                            min_value=0, max_value=3, value=0, step=1)
    if KitchenQual == 0:
        st.write(f"Selected KitchenQual Value: Ex (Excellent)")
    elif KitchenQual == 1:
        st.write(f"Selected KitchenQual Value: Fa (Fair)")
    elif KitchenQual == 2:
        st.write(f"Selected KitchenQual Value: Gd (Good)")
    else:
        st.write(f"Selected KitchenQual Value: TA (Typical/Average)")

    st.write('---')

    OverallCond  = st.slider("Select the overall condition rate of the house on a scale from 1 (Very Poor) to 10 (Very Excellent)", 
                             min_value=1, max_value=10, value=0, step=1)
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
    elif YearBuilt == 1 :
        st.write(f"Selected YearBuilt Value: 1881-1890")
    elif YearBuilt == 2 :
        st.write(f"Selected YearBuilt Value: 1891-1900")
    elif YearBuilt == 3 :
        st.write(f"Selected YearBuilt Value: 1901-1910")
    elif YearBuilt == 4 :
        st.write(f"Selected YearBuilt Value: 1911-1920")
    elif YearBuilt == 5 :
        st.write(f"Selected YearBuilt Value: 1921-1930")
    elif YearBuilt == 6 :
        st.write(f"Selected YearBuilt Value: 1931-1940")
    elif YearBuilt == 7 :
        st.write(f"Selected YearBuilt Value: 1941-1950")
    elif YearBuilt == 8 :
        st.write(f"Selected YearBuilt Value: 1951-1960")
    elif YearBuilt == 9 :
        st.write(f"Selected YearBuilt Value: 1961-1970")
    elif YearBuilt == 10 :
        st.write(f"Selected YearBuilt Value: 1971-1980")
    elif YearBuilt == 11 :
        st.write(f"Selected YearBuilt Value: 1981-1990")
    elif YearBuilt == 11 :
        st.write(f"Selected YearBuilt Value: 1991-2000")
    else :
        st.write(f"Selected YearBuilt Value: 2001-2010")

    st.write('---')

    YearRemodAdd = st.slider("Select the decade of remodeling or additions, with the same value as the construction date if no remodeling occurred", 
                             min_value=0, max_value=6, value=0, step=1)
    if YearRemodAdd == 0 :
        st.write(f"Selected YearRemodAdd Value: 1941-1950")
    elif YearRemodAdd == 1 :
        st.write(f"Selected YearRemodAdd Value: 1951-1960")
    elif YearRemodAdd == 2 :
        st.write(f"Selected YearRemodAdd Value: 1961-1970")
    elif YearRemodAdd == 3 :
        st.write(f"Selected YearRemodAdd Value: 1971-1980")
    elif YearRemodAdd == 4 :
        st.write(f"Selected YearRemodAdd Value: 1981-1990")
    elif YearRemodAdd == 5 :
        st.write(f"Selected YearRemodAdd Value: 1991-2000")
    else :
        st.write(f"Selected YearRemodAdd Value: 2001-2010")

    st.write('---')

    if st.button("Predict Sale Price"):
        df2 = pd.read_csv("for_modeling.csv")
        X = df2.drop("SalePrice", axis=1)
        y = df2["SalePrice"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)
        xgb.fit(X_train, y_train)

        user_values = [_1stFlrSF, GarageArea, GrLivArea, TotalBsmtSF, BedroomAbvGr, BsmtExposure, BsmtFinType1,
                       GarageFinish, GarageYrBlt, KitchenQual, OverallCond, OverallQual, YearBuilt, YearRemodAdd]
        user_input_df = pd.DataFrame([user_values], columns=X_train.columns)
        prediction = xgb.predict(user_input_df)
        st.success(f"Predicted Sale Price: ${prediction[0]:,.0f}")

