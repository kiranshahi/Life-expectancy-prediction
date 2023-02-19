# Prediction of life expectancy


### Introduction 
*“Life expectancy declining in many English communities even before pandemic”* (Head, 2021). The article published on Imperial College London’s blog grabbed our attention while researching for the dataset. After doing further research on this topic, we found that this is one of the alarming issues in developed countries like the UK and the US (Head, 2021). So, as a team, we decided to implement the skills that we had learned during our course to identify and understand the factors that can affect the life expectancy of human beings.

Based on the above problem statement, we designed the following research question.

### Research question:
> “Can we predict the life expectancy at birth based on the world development indicator such as Unemployment rate, Infant Mortality Rate, GDP, GNI, Clean fuels and cooking technologies, etc.,”

### Datasets:
In this research we have downloaded 4 different raw dataset from *World Bank* and *World Health Organization*. These datasets are as follows:

| Dataset  | Description |
| ------------- | ------------- |
| [World Development Indicators](https://databank.worldbank.org/data/download/WDI_csv.zip)  | This dataset contains the data of 1444 development indicators for 2666 countries and country groups between the years 1960 to 2020. This dataset was downloaded from the world bank’s data hub.  |
| [Health workforce](https://apps.who.int/gho/data/node.main.HWFGRP_0020?lang=en)  | This dataset contains the health workforce information such as medical doctors (per 10000 population), number of medical doctors, number of Generalist medical practitioners, etc.  |
| [Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)](https://data.worldbank.org/indicator/SH.DYN.NCOM.ZS) | This dataset contains information on mortality caused by various non-communicable diseases such as cardiovascular disease (CVD), cancer, diabetes etc. We have used two files for this dataset. Separately for both males and females. This dataset was downloaded from the world bank’s databank. |
| [Suicide morality rate (per 100,000 population)](https://data.worldbank.org/indicator/SH.STA.SUIC.P5) | This data set contains information on the suicide mortality rate per 100,000 population. We have used two files for this dataset. Separately for both males and females. This dataset was downloaded from the world bank’s databank.  |


> Complete and cleaned [Life Expectancy Dataset is publised in Kaggle](https://www.kaggle.com/datasets/kiranshahi/life-expectancy-dataset).

### Procedure
- **Data acquisition:** We have collected aforementioned public datasets from various sources.
- **Data preparation and cleaning:** We cleaned, merged and feature engineered (Principal Component Analysis - PCA) the datasets using [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), [missingno](https://github.com/ResidentMario/missingno) and [scikit learn](https://scikit-learn.org/stable/)
- **Exploratory data analysis:** We did Univariate Analysis by plotting each numerical variable on a histogram and boxplot to understand data distribution and outliers. Similarly, in Multivariate Analysis we plotted life expectancy against other numerical variables in a scatter plot to know the relationship between the variables. And dimensionality reduction using Unsupervised Learning technique.
- **Machine Learning Prediction:** For machine learning prediction, we have implemented Support Vector Machine (SVM), Random Forest, Decision Tree and K-Nearest Neighbour (KNN) for our research. Howevere, in this repository we have pubished only random forest algorithm.
- **Deep learning prediction:** Similarly, in deep learning method we have experimented various hyperparameters like hidden layers, activation funciton, optimizer and epochs.
- **Hyperparameter Tuning:** To tune the parameters in machine learning we have implemented [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) and in neural network we used [Keras Tuner](https://keras.io/api/keras_tuner/).
- **Evaluation of impact of features on model:** We plotted the [Shap value](https://github.com/slundberg/shap) to understand the impact of each features in model in deep learning. And in machine learning we plotted [features_importances_](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) property of [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) using [scikit learn](https://scikit-learn.org/stable/).

Complete report is available [here at github.](report.pdf)

### screenshots:

##### Exploratory Data Analysis (EDA)
![Male vs Female life expectancy](/images/Male-vs-Female-expected-life.png "Male vs Female life expectancy")

Findings from preliminary EDA: Female has higher life expectancy compared with male.
___

![Corelation Plot](/images/corelation-plot.png "Corelation Plot")

Corelation plot showing the multicollinearity issues among variables.

![Biplot between Component and variable](/images/pca-plot.png "Biplot between Component and variable")

Biplot showing the relationship between Principal Component and variable

![Visualization of variance and component in scree plot](/images/pca1-vs-pca2.png "Visualization of variance and component in scree plot")

Visualization of variance and component (PCA1 and PCA2) in scree plot

##### Data preparation and cleaning

![Missing and invalid data](/images/initial-missing-values.png "Missing and invalid data")

Visualization of missing and invalid data before cleaning.

![Missing and invalid data](/images/initial-missing-values-missingo.png "Missing and invalid data")

Visualization of [missing](https://github.com/ResidentMario/missingno) and invalid data before cleaning using missingno library.

![Missing and invalid data](/images/after-data-cleaning.png "Missing and invalid data")

Visualization of missing and invalid data after cleaning.

![Missing and invalid data](/images/after-data-cleaning-1.png "Missing and invalid data")

Visualization of missing and invalid data after cleaning using missingno library.

##### Performance of Machine Learning Algorithm in our dataset

![Machine learning performance](/images/random-forest-after-fine-tuning.png "Machine learning performance")

Performance of Random Forest after fine tuning of hyperparameters on Life expectancy datasets.

![Feature importance on model prediction](/images/Feature-importances-Random-forest.png "Feature importance on model prediction")

Visualization of feature importance on prediction. *#Explainable AI*

##### Performance of Deep Learning Algorithm (Neural Network) in our dataset

![Neural Network's performance](/images/model-performance-after-fine-tuning.png "Neural Network's performance")

Performance of Neural Network after fine tuning of hyperparameters on Life expectancy datasets.

![Visualization of shap value](/images/Feature-impact-on-prediction-result.png "Neural Network's performance")

Visualization of shap value on model's prediction. *#Explainable AI*

### References
Head, E., 2021. Life expectancy declining in many English communities even before pandemic | Imperial News | Imperial College London. [online] Imperial News. Available at: https://www.imperial.ac.uk/news/231119/life-expectancy-declining-many-english-communities/ [Accessed 23 March 2022].
