# Sentiment Analysis

The application predicts the sentiment of the given text based on news text provided to it.

The scope of this project can be extended to great limits.
(At present only news is available)
## Features

- Enter the text to be analyzed
- Use the topic of which the text is entered using the drop down menu
- Click on predict

First data is extracted data from the various sites using web scrapping and then the data is cleaned for any outlier text. The data is then classified using VADER's compound score and labeled as positive, negative or neutral.

This data is then used to train a model using Naive Bayes classifier.

![image](https://user-images.githubusercontent.com/73440161/136678526-2d304a20-4b11-4c4a-8a54-4b594b2cfc5d.png)

![image](https://user-images.githubusercontent.com/73440161/136678531-c0e59c5a-db49-4ac1-b3d9-77af115a831e.png)

![image](https://user-images.githubusercontent.com/73440161/136678535-cbf2b367-98bf-421f-82b1-2f820015ca94.png)

## Methodology
![image](https://user-images.githubusercontent.com/73440161/136678608-6eb58bd0-0f84-4cb7-87e8-d508611a1957.png)
