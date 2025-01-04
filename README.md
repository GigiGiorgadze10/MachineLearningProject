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


## API and Libraries

The project relies on the following libraries:

  - **requests**: For making HTTP requests to fetch car listing data from the website.
  - **beautifulsoup4**: For parsing and extracting relevant content from the HTML.
  - **csv**: For saving the scraped data in CSV format.
  - **re**: For handling regular expressions, used in cleaning and extracting relevant data.
  - **pandas**: For data manipulation, cleaning, and preprocessing.
  - **matplotlib**: For generating basic visualizations such as histograms, scatter plots, and bar charts.
  - **seaborn**: For generating more advanced visualizations such as pie charts and residual plots.
  - **scikit-learn**: For machine learning tasks such as model training (Random Forest Regressor) and evaluation (mean absolute error, mean squared error, etc.).

---

## Outputs

The project generates the following outputs:

  - **Scraped Data**: The car listings data is saved in a CSV file (`ss_cars.csv`), containing details such as car brand, model, year, mileage, transmission type, engine type, price, and location.
  - **Visualizations**: Various visualizations are generated to explore the data:
    - **Price Distribution Histogram**: Displays the distribution of car prices.
    - **Top Car Brands Bar Chart**: Visualizes the most common car brands in the dataset.
    - **Transmission Type Pie Chart**: Shows the distribution of transmission types in the listings.
    - **Top Locations Bar Chart**: Displays the top 10 locations where the cars are listed.
    - **Residual Plot**: Helps to assess how well the regression model's predictions match the actual prices.
    - **Scatter Plot**: Shows the comparison between actual prices and predicted prices.

---
