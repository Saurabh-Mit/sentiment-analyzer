# Sentiment Analysis

The application predicts the sentiment of the given text based on certain topics

- Oil Price
- Stock Markets

The scope of this project can be extended to great limits.
(At present only Oil Price is available)
## Features

- Enter the text to be analyzed
- Use the topic of which the text is entered using the drop down menu
- Click on predict

First data is extracted data from the various sites using web scrapping and then the data is cleaned for any outlier text. The data is then classified using VADER's compound score and labeled as positive, negative or neutral.

This data is then used to train a model using Naive Bayes classifier.
