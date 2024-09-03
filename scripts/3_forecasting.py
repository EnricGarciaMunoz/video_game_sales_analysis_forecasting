# Data Forecasting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.preprocessing import PolynomialFeatures # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load processed Data
data = pd.read_csv('../data/processed/vgsales_processed.csv')

columns = ['Year', 'Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

# We create the correlation matrix
corr_ = data[columns].corr()

# Create the heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(corr_, annot=True, linewidths=.2, cmap='RdYlBu_r')

plt.title('Correlation Matrix')
plt.show()

# First, we create dummy variables for 'Platform'
platform_dummies = pd.get_dummies(data['Platform'], prefix='Platform')

# Select the relevant column for analysis
data_with_dummies = pd.concat([data, platform_dummies], axis=1)
corr_data = data_with_dummies[['Global_Sales'] + list(platform_dummies.columns)]

# We create the correlation matrix
corr_ = corr_data.corr()

cmap = mcolors.LinearSegmentedColormap.from_list('white_to_red', ['white', 'darkred'])

# Create the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(corr_, annot=True, linewidths=.5, cmap=cmap, fmt='.2f')

plt.title('Platform vs. total Sales correlation')
plt.show()

# First, we create dummy variables for 'Genre'
platform_dummies = pd.get_dummies(data['Genre'], prefix='Genre')

# Select the relevant column for analysis
data_with_dummies = pd.concat([data, platform_dummies], axis=1)
# Seleccionar las columnas relevantes para la correlación
corr_data = data_with_dummies[['Global_Sales'] + list(platform_dummies.columns)]

# We create the correlation matrix
corr_ = corr_data.corr()
cmap = mcolors.LinearSegmentedColormap.from_list('white_to_red', ['white', 'darkred'])

# Create the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(corr_, annot=True, linewidths=.5, cmap=cmap, fmt='.2f')

plt.title('Genre vs. total Sales correlation')
plt.show()

top_publishers = data.groupby('Publisher')['Global_Sales'].sum().nlargest(5).index

# Filter the TOP 5 publishers
filtered_data = data[data['Publisher'].isin(top_publishers)]

# We create dummy variables for 'Publishers'
publisher_dummies = pd.get_dummies(filtered_data['Publisher'], prefix='Publisher')

# Select the relevant column for analysis
filtered_data_with_dummies = pd.concat([filtered_data[['Global_Sales']], publisher_dummies], axis=1)

# We create the correlation matrix
corr_ = filtered_data_with_dummies.corr()
cmap = mcolors.LinearSegmentedColormap.from_list('white_to_red', ['white', 'darkred'])

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_, annot=True, linewidths=.5, cmap=cmap, fmt='.2f')

plt.title('Publishers vs. total Sales correlation')
plt.show()

# Forecasting

from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore

# TOP 5 values per variable
top_platforms = data.groupby('Platform')['Global_Sales'].sum().nlargest(5).index
top_genres = data.groupby('Genre')['Global_Sales'].sum().nlargest(5).index
top_publishers = data.groupby('Publisher')['Global_Sales'].sum().nlargest(5).index

filtered_data = data[data['Platform'].isin(top_platforms) & 
                     data['Genre'].isin(top_genres) &
                     data['Publisher'].isin(top_publishers)]

X = filtered_data[['Platform', 'Genre', 'Publisher']]
y = filtered_data[['Global_Sales']]

# Create the Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre', 'Publisher'])
    ]
)

# We create the model. We'll be using a n_estimator of 100, given that we used just the top features, 
# and a random state of 32 (no matter what we have chosen)
model = RandomForestRegressor(n_estimators=100, random_state=32)

# We create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Then, we adjust the model with filtered data
pipeline.fit(X, y.values.ravel())

# We create a DataFrame  with all possible combinations of features
combinations = pd.MultiIndex.from_product([top_platforms, top_genres, top_publishers],
                                         names=['Platform', 'Genre', 'Publisher']).to_frame(index=False)

# We make predictions for all feature combinations
predictions = pipeline.predict(combinations)

combinations['Global_Sales'] = predictions

# Expected sales depending on Publisher and Genre
plt.figure(figsize=(14, 8))
pivot_table = combinations.pivot_table(index='Publisher', columns='Genre', values='Global_Sales')
sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, linewidths=.5)
plt.title('Total Sales prediction depending on Genre and Publisher')
plt.xlabel('Genre')
plt.ylabel('Publisher')
plt.show()

# Expected sales depending on Publisher and Genre
plt.figure(figsize=(14, 8))
pivot_table = combinations.pivot_table(index='Platform', columns='Genre', values='Global_Sales')
sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, linewidths=.5)
plt.title('Total Sales prediction depending on Genre and Platform')
plt.xlabel('Genre')
plt.ylabel('Platform')
plt.show()

top_platforms = data.groupby('Platform')['EU_Sales'].sum().nlargest(5).index
top_genres = data.groupby('Genre')['EU_Sales'].sum().nlargest(5).index
top_publishers = data.groupby('Publisher')['EU_Sales'].sum().nlargest(5).index

filtered_data = data[data['Platform'].isin(top_platforms) & 
                     data['Genre'].isin(top_genres) &
                     data['Publisher'].isin(top_publishers)]


X = filtered_data[['Platform', 'Genre', 'Publisher']]
y = filtered_data[['EU_Sales']]


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre', 'Publisher'])
    ]
)


model = RandomForestRegressor(n_estimators=100, random_state=32)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])


pipeline.fit(X, y.values.ravel())


combinations = pd.MultiIndex.from_product([top_platforms, top_genres, top_publishers],
                                         names=['Platform', 'Genre', 'Publisher']).to_frame(index=False)


predictions = pipeline.predict(combinations)


combinations['EU_Sales'] = predictions


plt.figure(figsize=(14, 8))
pivot_table = combinations.pivot_table(index='Publisher', columns='Genre', values='EU_Sales')
sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, linewidths=.5)
plt.title('EU Sales prediction depending on Genre and Publisher')
plt.xlabel('Genre')
plt.ylabel('Publisher')
plt.show()

plt.figure(figsize=(14, 8))
pivot_table = combinations.pivot_table(index='Platform', columns='Genre', values='EU_Sales')
sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, linewidths=.5)
plt.title('EU Sales prediction depending on Genre and Platform')
plt.xlabel('Genre')
plt.ylabel('Platform')
plt.show()

top_platforms = data.groupby('Platform')['JP_Sales'].sum().nlargest(5).index
top_genres = data.groupby('Genre')['JP_Sales'].sum().nlargest(5).index
top_publishers = data.groupby('Publisher')['JP_Sales'].sum().nlargest(5).index

filtered_data = data[data['Platform'].isin(top_platforms) & 
                     data['Genre'].isin(top_genres) &
                     data['Publisher'].isin(top_publishers)]


X = filtered_data[['Platform', 'Genre', 'Publisher']]
y = filtered_data[['JP_Sales']]


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre', 'Publisher'])
    ]
)


model = RandomForestRegressor(n_estimators=100, random_state=32)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])


pipeline.fit(X, y.values.ravel())


combinations = pd.MultiIndex.from_product([top_platforms, top_genres, top_publishers],
                                         names=['Platform', 'Genre', 'Publisher']).to_frame(index=False)


predictions = pipeline.predict(combinations)


combinations['JP_Sales'] = predictions


plt.figure(figsize=(14, 8))
pivot_table = combinations.pivot_table(index='Publisher', columns='Genre', values='JP_Sales')
sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, linewidths=.5)
plt.title('JP Sales prediction depending on Genre and Publisher')
plt.xlabel('Genre')
plt.ylabel('Publisher')
plt.show()


plt.figure(figsize=(14, 8))
pivot_table = combinations.pivot_table(index='Platform', columns='Genre', values='JP_Sales')
sns.heatmap(pivot_table, cmap='RdYlBu_r', annot=True, linewidths=.5)
plt.title('EU Sales prediction depending on Genre and Platform')
plt.xlabel('Genre')
plt.ylabel('Platform')
plt.show()

annual_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()

model = ExponentialSmoothing(annual_sales['Global_Sales'], trend='add', seasonal=None, seasonal_periods=None)
model_fit = model.fit()

# Predicting the next 10 years
future_years = np.arange(2016, 2026)
future_forecast = model_fit.forecast(len(future_years))

# Create a Data Frame for future predictions
future_sales = pd.DataFrame({'Year': future_years, 'Global_Sales': future_forecast})
future_sales = np.maximum(future_sales, 0)


plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Global_Sales', data=annual_sales, label='Past Data')
sns.lineplot(x='Year', y='Global_Sales', data=future_sales, label='Predictions', linestyle='--')
plt.title('Total Sales Forecasting over the next 10 years')
plt.xlabel('Year')
plt.ylabel('Global Sales (millions)')
plt.legend()
plt.show()

annual_sales = data.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()


X = annual_sales['Year'].values.reshape(-1, 1)
y = annual_sales[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].values


poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)


future_years = np.arange(2016, 2026).reshape(-1, 1)
future_years_poly = poly.transform(future_years)
future_sales = model.predict(future_years_poly)


future_sales = np.maximum(future_sales, 0)


future_data = pd.DataFrame({
    'Year': future_years.flatten(),
    'NA_Sales': future_sales[:, 0],
    'EU_Sales': future_sales[:, 1],
    'JP_Sales': future_sales[:, 2],
    'Other_Sales': future_sales[:, 3]
})


colors = {
    'NA_Sales': 'blue',
    'EU_Sales': 'green',
    'JP_Sales': 'red',
    'Other_Sales': 'orange'
}


plt.figure(figsize=(14, 8))
for region in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']:
    plt.plot(annual_sales['Year'], annual_sales[region], label=f'Actual {region}', color=colors[region])
    plt.plot(future_data['Year'], future_data[region], linestyle='--', color=colors[region], label=f'Predicted {region}')
plt.title('Sales Forecasting per Region')
plt.xlabel('Year')
plt.ylabel('Sales (milions)')
plt.legend()
plt.show()