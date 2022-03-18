**Disclaimer:** We read a lot of "how to READMEs" but were genuinely very confused by a lot of the information there and were not sure how to do it properly. So we just decided to do a written explanation to make it a little easier for you to navigate our repository! 

# The carbon foodprint of produce :apple: :blueberries: :pear:

We decided that we would like to create a CO2 Score for food and run a machine learning algorithm over our data to sort it into five categories. We wanted to do something similiar to the NutriScore and our goal was to sort different food products into the categories based on their carbon emissions.

## Project Outline

We quickly realised that we would not be able to realise our original idea which made us decide to downsize our project. We made a decision to only look at fresh fruits and vegetables.

We started collecting data and came across quite a few obstacles. After a few weeks of barely making any progess, we had to admit defeat. There was not really any data around to collect. During a couple of brainstorming sessions we created a miro board and mapped out our goals and what kind of data we would need to reach them. Then we decided to calculate our own Co2 Score based on different parameters.

We wanted to be able to calculate a Co2 Score for a piece of fruit depending on:

* its origin
* the transport type
* the time it is bought
* whether it is organic or not

For example we used coordinates of airports and harbours to calculate the distance between the origin country and Germany and then used an average Co2 value that describes how much Co2 per KM is emitted by plane/ship transportation. 

In the end we managed to clean our data and calculate all the carbon emissions that are emmited in each step. Then we started trying out different machine learning models until deciding on Random Forest and optimizing and fine tuning that model.

## Data

* **airport_data.xlsx**

:arrow_right: Airport coordinates

* **worldwide_port_data.xlsx**

:arrow_right: Port/Harbour coordinates

* **fruit_veggies_agriculture_base_CO2.xlsx**

:arrow_right: Base Co2 value

* **Rewe Web Scraping Produce Countries (incomplete).ipynb**

:arrow_right: First tries of web scraping back when we still wanted to have common origin countries for specific produce items

* **calc_data_final.xlsx**

:arrow_right: Final spreadsheet with calculated data 

* **main.py**

:arrow_right: Our code

## Authors :woman_technologist: :technologist:

* Daria Lysenko
* Vivienne Simunec
* Sebastian Leszinski
