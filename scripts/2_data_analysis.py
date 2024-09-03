import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar datos procesados
data = pd.read_csv('../data/processed/vgsales_processed.csv')


# Sales Distribution Over Years

# Total Sales Analysis per year
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Global_Sales', data=data.groupby('Year')['Global_Sales'].sum().reset_index())
plt.title('Global Sales per Year')
plt.xlabel('Year')
plt.ylabel('Total Sales (milions)')
plt.show()


# Group Data per Year and sum the total Sales
yearly_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()

# Create the Histogram
plt.figure(figsize=(14, 8))
sns.histplot(yearly_sales, x='Year', weights='Global_Sales', bins=yearly_sales['Year'].nunique(), kde=True)
plt.title('Total Sales per Year Histogram')
plt.xlabel('Year')
plt.ylabel('Total Sales (milions)')
plt.show()

# We work with sales data gruped by region
region_sales = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

# Calculate the percentages
region_sales_percent = region_sales / region_sales.sum() * 100

# Create a Donut Chart
plt.figure(figsize=(10, 6))
plt.pie(region_sales_percent, labels=region_sales.index, autopct='%1.1f%%', startangle=140)
plt.title('Sales percentages by Region')
plt.axis('equal')
plt.show()

# Sales evolution per region
regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
plt.figure(figsize=(14, 8))
for region in regions:
    sns.lineplot(x='Year', y=region, data=data.groupby('Year')[region].sum().reset_index(), label=region)
plt.title('Sales evolution per Region')
plt.xlabel('Year')
plt.ylabel('Sales (Millions)')
plt.legend()
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(y='Platform', data=data, order=data['Platform'].value_counts().index)
plt.title('Games Released per Platform')
plt.xlabel('Total Games Released')
plt.ylabel('Platform')
plt.show()

platform_sales = data.groupby('Platform')['Global_Sales'].sum().reset_index()

# Sort values
platform_sales = platform_sales.sort_values(by='Global_Sales', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(y='Platform', x='Global_Sales', data=platform_sales, palette='viridis', hue='Platform', legend=False)
plt.title('Sales per Platform')
plt.xlabel('Total Sales (Millions)')
plt.ylabel('Platform')
plt.show()

# Identify the TOP 5 Platforms in terms of sales
top_platforms = data.groupby('Platform')['Global_Sales'].sum().nlargest(5).index

# We create a chart for each Platform
regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
for platform in top_platforms:
    platform_data = data[data['Platform'] == platform].groupby('Year')[regions].sum().reset_index()
    
    plt.figure(figsize=(7, 4))
    for region in regions:
        sns.lineplot(x='Year', y=region, data=platform_data, label=region)
    plt.title(f'Sales Evolution per Region in Platform {platform}')
    plt.xlabel('Year')
    plt.ylabel('Sales (Millions)')
    plt.legend(title='Region')
    plt.show()

    # Identify and Delete rows with 'DS' as Platform in years before 2004
data = data[~((data['Platform'] == 'DS') & (data['Year'] < 2004))]

# We save changes on our csv
data.to_csv('../data/processed/vgsales_processed_filtered.csv', index=False)

# Verify that data was deleted.
print(data[(data['Platform'] == 'DS') & (data['Year'] < 2004)])

filtered_data = data[data['Platform'].isin(top_platforms)]

platform_sales_by_year = filtered_data.groupby(['Year', 'Platform'])['Global_Sales'].sum().reset_index()

plt.figure(figsize=(14, 8))

colors = sns.color_palette('husl', len(top_platforms))

for idx, platform in enumerate(top_platforms):
    platform_data = platform_sales_by_year[platform_sales_by_year['Platform'] == platform]
    plt.fill_between(platform_data['Year'], platform_data['Global_Sales'], color=colors[idx], alpha=0.6, label=platform)

plt.title('Sales Evolution for the 5 Platforms with most sales')
plt.xlabel('Year')
plt.ylabel('Total Sales (Millions)')
plt.legend(title='Platform')
plt.show()

# Game release distribution per genre
plt.figure(figsize=(14, 8))
sns.countplot(y='Genre', data=data, order=data['Genre'].value_counts().index)
plt.title('Game Release distribution per genre')
plt.xlabel('Amount of Games Released')
plt.ylabel('Genre')
plt.show()

# Sales per Platform
genre_sales = data.groupby('Genre')['Global_Sales'].sum().reset_index()

genre_sales = genre_sales.sort_values(by='Global_Sales', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(y='Genre', x='Global_Sales', data=genre_sales, palette='viridis', hue='Genre', legend=False)
plt.title('Total Sales per Genre')
plt.xlabel('Total Sales (Millions)')
plt.ylabel('Genre')
plt.show()

# Identif5 TOP 5 Genres based on Total Sales
top_genres = data.groupby('Genre')['Global_Sales'].sum().nlargest(5).index

regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
for genre in top_genres:
    genre_data = data[data['Genre'] == genre].groupby('Year')[regions].sum().reset_index()
    
    plt.figure(figsize=(7, 4))
    for region in regions:
        sns.lineplot(x='Year', y=region, data=genre_data, label=region)
    plt.title(f'Sales Evolution per Region in genre {genre}')
    plt.xlabel('Year')
    plt.ylabel('Sales (Millions)')
    plt.legend(title='Region')
    plt.show()

    # We'll analyze the TOP 10 companies in terms of sales
top_publishers_sales = data.groupby('Publisher')['Global_Sales'].sum().nlargest(10).reset_index()
publisher_sales = data.groupby('Publisher').agg({'Global_Sales': 'sum', 'Name': 'count'}).reset_index()
publisher_sales.rename(columns={'Name': 'Game_Count'}, inplace=True)
top_publishers = publisher_sales.nlargest(10, 'Global_Sales')

filtered_data = data[data['Publisher'].isin(top_publishers)]

top_publishers_sales = top_publishers_sales.sort_values(by='Global_Sales', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(y='Publisher', x='Global_Sales', data=top_publishers_sales, palette='viridis', hue='Publisher', legend=False)
plt.title('Total Sales per publisher')
plt.xlabel('Total Sales (Millions)')
plt.ylabel('Publisher')
plt.show()

# Calculate total sales and game count by Publisher
publisher_sales = data.groupby('Publisher').agg({'Global_Sales': 'sum', 'Name': 'count'}).reset_index()
publisher_sales.rename(columns={'Name': 'Game_Count'}, inplace=True)

# Identify the 10 Publishers with the most global sales
top_publishers = publisher_sales.nlargest(10, 'Global_Sales')

# Calculate the sales ratio per game for the 10 publishers
top_publishers['Sales_per_Game'] = top_publishers['Global_Sales'] / top_publishers['Game_Count']

# Create the bar chart
plt.figure(figsize=(14, 8))
sns.barplot(x='Sales_per_Game', y='Publisher', data=top_publishers, palette='viridis', hue='Publisher', legend=False)
plt.title('Sales ratio for top 10 Publishers')
plt.xlabel('Sales per Game (milions)')
plt.ylabel('Publisher')
plt.show()