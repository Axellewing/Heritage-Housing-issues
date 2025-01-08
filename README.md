# PROJECT - Heritage Housing Insights

## Overview

This project aims to provide comprehensive insights and predictions related to house prices in Ames, Iowa, USA. I will deveelop a reliable predictive model for house prices in AMes, Iowa. 
Enabling Lydia to estimate the sales price for her inherited properties. My goal is to enhance Lydia's understanding of the Ames, Iowa housing maket, accounting for factors that contribute
to a house's desirability and value. 

## Table of Contents

- [CRISP-DM Methodology](#crisp-dm-methodology)
  - [Business Understanding](1.business-understanding)
  - [Data Understanding](2.data-understanding)
  - [Data Preparation](3.data-preparation)
  - [Modeling](4.modeling)
  - [Evaluation](5.evaluation)
  - [Deployment](6.deployment)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [Business Requirements](#business-requirements)

## CRISP-DM Methodology

## 1. Business Understanding

In this step, we describe the business problem or question the project aims to address, emphasizing key challenges and goals related to housing issues in Ames.

### Business Requirements:
* Requirements for Data Visualization:
    * Visual representations of housing price trends over time.
    * Interactive visualizations showcasing the correlation between various features and sales prices.
* Requirements for Machine Learning Tasks:
    * Accurate house price predictions for Lydia's properties, enabling informed decision-making.
* User Stories:
    * As a user, I want to visualize historical house prices to understand market trends.
    * As a user, I want to input features and receive accurate price predictions to evaluate my inherited properties.

### 2. Data Understanding

- I explain the data sources used, including the dataset from Kaggle, which comprises 1,460 rows with various features that describe the properties. The dataset includes aspects such as the number of bedrooms, square footage, year built, etc.
  I discuss relevant attributes and their significance in predicting house prices.

### 3. Data Preparation

The steps taken to clean, preprocess, and transform the raw data for analysis include:
- Imputation.
- Encoding.
- Binning.
- Standardization.

### 4. Modeling

We train various algorithms such as Linear Regression, Polynomial Regression, SVM, XGBoost, ...

### 5. Evaluation

We discuss the R2 score and select the best model.

### 6. Deployment

- Develop a dashboard using the Streamlit framework to provide a user-friendly interface for utilizing the model.
- Deploy the app on Heroku to make it accessible online.

## Project Structure

- **app.py**: Main file containing the Streamlit app.
- **requirements.txt**: File listing the required dependencies.
- **Procfile**: Configuration file for deploying the app on Heroku.
- **runtime.txt**: Specifies the Python runtime version for Heroku.
- **setup.sh**: Shell script for setting up necessary configurations.

## Requirements

To run the app locally, create a virtual environment (venv) and install the dependencies inside it.

## Usage

To run the app locally, use the following command: `streamlit run app.py`.

## Acknowledgments

The dataset was downloaded from Kaggle, and the project was developed following the CRISP-DM methodology.

## Business Requirements

* Data Visualization: Focuses on presenting data insights and trends to better inform users about the housing market.
* Machine Learning: Targets the development of robust predictive models to guide property valuation processes.
