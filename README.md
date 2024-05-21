# Predictive Analytics for Car Pricing

## Overview
This project aims to analyze automotive advertisements from the Otomoto platform to identify key factors affecting car prices and develop a model for predicting the prices of listed vehicles.

## Objectives
- Collect a dataset of car ads with relevant attributes such as make, model, year, mileage, condition, and listed price.
- Analyze the dataset to identify key factors significantly affecting car prices.
- Develop and train a machine learning model capable of predicting car prices based on the identified factors.
- Create a tool that evaluates car ads for potential under- or overpricing based on the model's predictions.

## Repository Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for data collection, cleaning, EDA, and modeling.
- `scripts/`: Python scripts for various tasks such as data scraping, preprocessing, and model training.
- `app/`: Code for the prediction tool application.
- `.gitignore`: List of files and directories to ignore in the repository.
- `requirements.txt`: List of dependencies.
- `README.md`: Project overview and instructions.

## Setup Instructions
1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd Predictive-Analytics-Car-Pricing`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the data scraping script: `python scripts/scrape_data.py`
