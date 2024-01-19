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
    


