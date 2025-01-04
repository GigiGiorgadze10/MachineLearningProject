import requests
from bs4 import BeautifulSoup
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

base_url = "https://ss.ge/ka/auto/list/iyideba?AutoManufacturerId=&VehicleModelIds=&VehicleDealTypes=2&YearFrom=&YearTo=&CurrencyId=1&PriceFrom=&PriceTo=&CityIds=&Query=&MillageFrom=&MillageTo=&EngineFrom=&EngineTo=&IsCustomsCleared=&TechnicalInspection=&AutoDealershipId=&El_Windows=false&Hatch=false&CruiseControl=false&StartStopSystem=false&AUX=false&Bluetooth=false&AntiFogHeadlights=false&SeatHeater=false&SmartSeats=false&MultiWheel=false&Signalisation=false&BoardComputer=false&Conditioner=false&ClimateControl=false&RearViewCamera=false&Monitor=false&PanoramicCeiling=false&ParkingControl=false&ABS=false&CentralLocker=false&AutoParking=false&LedHeadlights=false&AdaptedForDisabled=false&page="

csv_file = 'ss_cars.csv'
csv_columns = ['Car Brand and Model', 'Year', 'Mileage', 'Transmission', 'Engine Type', 'Price', 'Location']

with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)

    for page_number in range(1, 6):
        url = base_url + str(page_number)
        response = requests.get(url)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        car_cards = soup.find_all('div', class_='cars-item')

        for card in car_cards:
            car_title = card.find('h4').text.strip()
            match_brand_model = re.search(r'[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?', car_title)
            brand_model = match_brand_model.group(0) if match_brand_model else "Unknown"
            match_year = re.search(r'\b(19|20)\d{2}\b', car_title)
            year = match_year.group(0) if match_year else "Unknown"

            mileage = card.find('div', class_='cars-details').find_all('div')[0].text.strip()
            transmission = card.find('div', class_='cars-details').find_all('div')[1].text.strip()
            engine_type = card.find('div', class_='cars-details').find_all('div')[2].text.strip()

            price = card.find('div', class_='item-price')
            if price:
                price = price.find('div', class_='priceGel')
                if price:
                    price = price.text.strip()
                else:
                    price = "Price Not Available"
            else:
                price = "Price Not Available"

            location_date = card.find('div', class_='time-loaction').text.strip().split(', ')
            location = location_date[0] if location_date else "Unknown"

            writer.writerow([brand_model, year, mileage, transmission, engine_type, price, location])

print(f"Data saved to {csv_file}")

df = pd.read_csv('ss_cars.csv')

df['Price'] = df['Price'].apply(lambda x: re.sub(r'[^\d]', '', str(x)))
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

df['Mileage'] = df['Mileage'].apply(lambda x: re.sub(r'\D', '', str(x)))
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

df = df.dropna(subset=['Price', 'Mileage'])

df['Car Brand'] = df['Car Brand and Model'].apply(lambda x: x.split(' ')[0])
df['Car Model'] = df['Car Brand and Model'].apply(lambda x: ' '.join(x.split(' ')[1:]))

X = df[['Car Brand', 'Car Model', 'Year', 'Mileage', 'Transmission', 'Engine Type', 'Location']]
y = df['Price']

numeric_features = ['Year', 'Mileage']
numeric_transformer = StandardScaler()

categorical_features = ['Car Brand', 'Car Model', 'Transmission', 'Engine Type', 'Location']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_pred.head())

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

best_rf_model = grid_search.best_estimator_

y_pred_best = best_rf_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)

print(f"Best Model - Mean Absolute Error: {mae_best}")
print(f"Best Model - Mean Squared Error: {mse_best}")

importances = best_rf_model.named_steps['regressor'].feature_importances_
columns = numeric_features + list(best_rf_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

feature_importance_df = pd.DataFrame({
    'Feature': columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

residuals = y_test - y_pred_best

plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=residuals, lowess=True, color='green', line_kws={'color': 'red'})
plt.title('Residual Plot')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7, color='blue')
plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

brand_counts = df['Car Brand and Model'].value_counts().head(10)
plt.figure(figsize=(10, 6))
brand_counts.plot(kind='bar', color='skyblue')
plt.title('Top 10 Car Brands in Listings')
plt.xlabel('Car Brand and Model')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
df['Price'].dropna().plot(kind='hist', bins=20, color='green', edgecolor='black')
plt.title('Price Distribution of Cars')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

transmission_counts = df['Transmission'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title('Distribution of Transmission Types')
plt.axis('equal')
plt.show()

location_counts = df['Location'].value_counts().head(10)
plt.figure(figsize=(10, 6))
location_counts.plot(kind='bar', color='lightcoral')
plt.title('Top 10 Locations of Cars')
plt.xlabel('Location')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
