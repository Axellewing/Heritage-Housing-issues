# PROJECT - Heritage Housing Insights

## Overview

This project aims to provide comprehensive insights and predictions related to house prices in Ames, Iowa, USA.

## Table of Contents

- [CRISP-DM Methodology](#crisp-dm-methodology)
  - [Business Understanding](#business-understanding)
  - [Data Understanding](#data-understanding)
  - [Data Preparation](#data-preparation)
  - [Modeling](#modeling)
  - [Evaluation](#evaluation)
  - [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## CRISP-DM Methodology

### Business Understanding

In this step, I describe the business problem or question the project aims to address, emphasizing key challenges and goals related to housing issues in Ames.

### Data Understanding

- I explain the data sources used, including their nature, structure, and any initial exploration performed. 
- I discuss relevant attributes and their significance in predicting house prices.

### Data Preparation

The steps taken to clean, preprocess, and transform the raw data for analysis include:
- Imputation.
- Encoding.
- Binning.
- Standardization.

### Modeling

I have trained various algorithms such as Linear Regression, Polynomial Regression, SVM, XGBoost, ...

### Evaluation

I discuss the R2 score and select the best model.

### Deployment

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
