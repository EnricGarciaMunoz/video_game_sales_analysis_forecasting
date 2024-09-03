# Video Game Sales Analysis and Sales Forecasting

![Video_Games](Images/video_games_sales.jpg)

### A Python Project about the crysis of sales evolution of videoGames Sales until 2016 and a Sales Forecasting.

Video Games sales had a really unexpected development on previous years. We know now that videogames is one of the most prolific sources of sales, but that wasn't the forecast many years before. Why wasn't that the forecast? What was the forecast at that time? In this project we'll give answers to these questions!

I've structured this project in different steps, in order to focus on the most important part on every step:

## Steps of the project

* Question Formulation
* Data Acquisition, Cleaning and Exploration
* Data Analysis
* Forecasting

Without further ado, let's begin with the main points:

### Question Formulation

As we may know, all the Data Projects rise from a question that needs to be solved. The questions on this project, given what we learned many lines before, are:

- Why was the videogames forecast so bad in 2016?

- What was the forecast at that time?


### Data Acquisition, Cleaning and Exploration

When trying to Aquire the Data, you know what they say: "Garbage in, garbage out!" That means that you need reliable data if you want reliable results. That is why we have chosen [vgchartz](https://www.vgchartz.com/gamedb/) for this projects. They have a database of the videoGames Sales, and are a well known and reliable website among the videogame lovers.

The data is stored in a csv file called "vgsales.csv" available on this projects files.

Just to mention, and interesting point to take into account is that we avoided data from 2016, because data was taken at some point that year, and the data wasn't complete, so we'll do analysis up to 2015.

- Games Released per Year without removing data
[Games_Release_per_Year](Images/Games_Released_per_Year.PNG)

- Games Released per Year up to 2015
[Games_Release_per_Year_2](Images/Games_Released_per_Year_2.PNG)


### Data Analysis

In this step I did a deep analysis on games sales from different perspectives, analysing every feature and columns of the dataset, even analysed per every Region. I encourage you to execute the Data_Analysis file (2_data_analysis.ipnb). I'll leave here some interesting analysis made on it, but there's many more with interesting insigths!

- Sales Evolution for the main Platforms
[Sales_Evolution_for_TOP5_Platforms](Images/Sales_Evolution_for_TOP5_Platforms.PNG)

- Sales Ratio per games released on Top 10 Publishers
[Sales_Ratio_per_TOP10_publishers](Images/Sales_Ratio_per_TOP10_publishers.PNG)

Some interesting analysis made on this step are the following:

- There was a general fall in sales in every region, that seemed to end with the video games sales anytime soon (further explenations on the file).

- Japan was the only Region with a differentiate behaviour on videogames buying. The other regions had a really similar behaviour, being North_America the region with the most sales.

- Nintendo seems to be the most lucrative company by far, compared to the others, and they are great in specific videogames genres.


### Forecasting

In this last step, we create an algorithm in order to forecast the future sales develop, and the sales expected per genres, publishers and platforms, combining this features on correlation matrixes.

In this step we can see pretty clearly how the sales were really expected to disappear in a few years.

- Sales Forecasting per Region
[Sales_Forecasting_per_Region](Images/Sales_Forecasting_per_Region.PNG)


## Conclusions

Finishing this projects, I've been able to solve the Questions purposed at the beginning. If it wasn't for a big change in videogames industry, it seems indeed that regular videogames sales were about to disappear! (at least on regular shops)


## Project file Structure

- `data/`: Contains the original and processed Data
- `notebooks/`: Contains the Jupyter Notebooks used to run the project.
- `Scripts/`: Contains the scripts of the project.
- `Images/`: Images and Charts resulting from the scripts.


## How to install and execute the project

- Clone the repository with the following command:

    ```bash
    git clone https://github.com/EnricGarciaMunoz/video_game_sales_analysis_forecast.git
    cd video-game-sales-analysis
    ```

-  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

-  Execute main.py:
    ```bash
    python scripts/data_processing.py
    python scripts/data_analysis.py
    python scripts/forecasting.py
    ```

This will execute the project. I really hope you find it interesting!