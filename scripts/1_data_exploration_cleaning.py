import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/raw/vgsales.csv')

print(data.head())

print(data.describe())

print(data.info())


# Data Cleaning

# Analyze rows with null values
print("Null values:")
print(data.isnull().sum())

# Analyzing rows with null values in 'Year' column
print("Rows with null values in 'Year' column:")
null_year = data[data['Year'].isnull()]
print(null_year.describe())

# Analyzing rows with null values in 'Publisher' column
print("Rows with null values in 'Publisher' column:")
null_publisher = data[data['Publisher'].isnull()]
print(null_publisher.describe())

# Delete rows with null values
data = data.dropna()

print(data.isnull().sum())

data['Year'] = pd.to_datetime(data['Year'], format='%Y').dt.year

print(data.info())


# Data Distribution Analysis

plt.figure(figsize=(10, 6))
sns.histplot(data['Global_Sales'], bins=60, kde=True)
plt.title('Global Sales Distribution')
plt.xlabel('Global Sales (Millions)')
plt.ylabel('Frquency')
plt.show()

# Games released per year
games_per_year = data['Year'].value_counts().sort_index()

# We create the histogram
plt.figure(figsize=(14, 8))
sns.barplot(x=games_per_year.index, y=games_per_year.values, color='skyblue')
plt.title('Games released per year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.show()

# Games sold per year
# First we group Data per year and sum the total of that year
yearly_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()

# We create the histogram
plt.figure(figsize=(14, 8))
sns.histplot(yearly_sales, x='Year', weights='Global_Sales', bins=yearly_sales['Year'].nunique())
plt.title('Global Sales per Year')
plt.xlabel('Year')
plt.ylabel('Global Sales (Milions)')
plt.show()


data = data[data['Year'] <= 2015]

# Games released per year
games_per_year = data['Year'].value_counts().sort_index()

# We create the histogram
plt.figure(figsize=(14, 8))
sns.barplot(x=games_per_year.index, y=games_per_year.values, color='skyblue')
plt.title('Games released per year (Up to 2015)')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.show()

# Games sold per year
# First we group Data per year and sum the total of that year
yearly_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()

# We create the histogram
plt.figure(figsize=(14, 8))
sns.histplot(yearly_sales, x='Year', weights='Global_Sales', bins=yearly_sales['Year'].nunique())
plt.title('Global Sales per Year (Up to 2015)')
plt.xlabel('Year')
plt.ylabel('Global Sales (Milions)')
plt.show()

# Guardar el DataFrame procesado en un archivo CSV
data.to_csv('../data/processed/vgsales_processed.csv', index=False)