# Machine Learning Project About cars From ss.ge

## Overview

This project is designed for scraping car listings from [SS.ge](https://ss.ge), cleaning and preprocessing the data, and performing machine learning tasks such as regression. It also includes visualizations for exploring the data and understanding trends related to car prices and other factors.

---

## Features

### Web Scraping
- Scrapes car listings from [SS.ge](https://ss.ge/ka/auto/list/iyideba), including:
  - Car name and model
  - Price
  - Year
  - Mileage
  - Transmission type
  - Engine type
  - Location
- Saves data to a CSV file (`ss_cars.csv`).

### Data Cleaning and Preprocessing
- Cleans price data to ensure numeric format.
- Standardizes mileage values.
- Encodes categorical variables (e.g., transmission type, engine type, car brand).
- Handles missing data by removing rows with essential missing values.
  
### Machine Learning Models
- **Random Forest Regressor**: Predicts car prices based on features such as mileage, year, transmission type, engine type, and location.
- **Hyperparameter Tuning**: Utilizes GridSearchCV to find the best parameters for the model and improve performance.

### Data Visualization
- Generates insightful visualizations:
  - Residual plot and scatter plot for regression analysis (actual vs. predicted prices).
  - Histograms for price distribution.
  - Bar charts for the top 10 car brands.
  - Pie charts for transmission type distribution.
  - Bar charts for the top 10 car locations.

---

## Installation

Follow the steps below to set up the project environment:

1. Clone the repository:
   ```bash
   https://github.com/GigiGiorgadze10/MachineLearningProject.git
2. Navigate to the project directory:
  ```bash
  cd MachineLearningProject
  ```
3. Install dependencies:
  ```
  pip install -r requirements.txt
  ```
