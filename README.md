**Disclaimer:** We read a lot of "how to READMEs" but were genuinely very confused by a lot of the information there and were not sure how to do it properly. So we just decided to do a written explanation to make it a little easier for you to navigate our repository! 

# The carbon foodprint of produce :apple: :blueberries: :pear:

We decided that we would like to create a CO2-score for food items and run different machine learning algorithms over our data to sort different fruit and vegetables into five different categories based on their climate impact (A-E). The climate score is heavily inspired by the Nutri-Score which gives an indicator on the nutritional value of different food items. 

## Project Outline

We quickly realised that we would not be able to realise our original idea of including all kinds of food categories. It made us downsize our project by only looking at fresh fruits and vegetables.

We started collecting data and came across quite a few obstacles. After a few weeks of barely making any progess, we had to admit defeat. There were not any concise datasets on CO2 emissions of fresh produce around to collect. During a couple of brainstorming sessions we created a miro board and mapped out our goals and what kind of data we would need to reach them. After that we decided to calculate our own CO2-score based on different parameters.

We wanted to be able to calculate a Co2 Score for a piece of fruit depending on:

* it's origin
* the transport type
* the month it is bought in
* whether it is organic or not
* whether it was grown in a greenhouse or not

For example, we used GPS coordinates of airports and harbours around the world to calculate the distance between the origin country of the produce and Germany and then used an estimate of the average CO2 amount emitted by plane/ship transportation.

In the end we managed to fully clean our data and calculate all the carbon emissions that are emmited in each step of the value chain. Then we started trying out different machine learning models until deciding on the Random Forest. This gave us an accuracy of roughly 87% and fine-tuning of the model lead to a final accuracy of 92%.

## Data

* **airport_data.xlsx**

:arrow_right: Airport coordinates

* **worldwide_port_data.xlsx**

:arrow_right: Port/Harbour coordinates

* **fruit_veggies_agriculture_base_CO2.xlsx**

:arrow_right: Base Co2 value

* **calc_data_final.xlsx**

:arrow_right: Final spreadsheet with calculated data (shows the data structure after data cleaning)

* **main.py**

:arrow_right: Our code (including data cleaning, data modeling and model evaluation)

* **Rewe Web Scraping Produce Countries (incomplete).ipynb**

:arrow_right: First tries of web scraping back when we still wanted to have common origin countries for specific produce items (not needed to run the python file)

## Required packages 📦
* numpy
* pandas
* geopy.distance
* matplotlib.pyplot
* seaborn
* sklearn


## Authors :woman_technologist: :technologist:

* Daria Lysenko
* Vivienne Simunec
* Sebastian Leszinski
