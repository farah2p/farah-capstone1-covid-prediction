# COVID-19 New Cases Prediction in Malaysia
## Dataset

The dataset used for training and testing the model is sourced from https://github.com/MoH-Malaysia/covid19-public. It includes daily COVID-19 case counts in Malaysia for a specific time period. The dataset consists of the following columns:

- date: The date of the recorded data
- cases_new: The number of new COVID-19 cases reported on that date.
- cases_new
- cases_import
- cases_recovered
- cases_active
- cases_cluster
- cases_unvax
- cases_pvax
- cases_fvax
- cases_boost
- cases_child
- cases_adolescent
- cases_adult
- cases_elderly
- cases_0_4
- cases_5_11
- cases_12_17
- cases_18_29
- cases_30_39
- cases_40_49
- cases_50_59
- cases_60_69
- cases_70_79
- cases_80
- cluster_import
- cluster_religious
- cluster_community
- cluster_highRisk
- cluster_education
- cluster_detentionCentre
- cluster_workplace

## Software
- Visual Studio Code
- TensorFlow framework

## Methodology
1) Problem Definition: The goal is to develop a predictive model that can forecast the number of new COVID-19 cases (cases_new) in Malaysia for future time steps based on the historical data of the number of cases over the past 30 days
2) Data Collection: Gather the historical data of the number of COVID-19 cases in Malaysia, including the daily counts of cases over the past 30 days. The dataset includes the dates and the corresponding number of cases_new for each date.
3) Data Preprocessing: Perform necessary preprocessing steps like KNNImputer to handle the missing values and MinMaxScaler for scaling the data.
4) LSTM Model Architecture: Design the LSTM neural network architecture for the prediction task. The model take the past 30 days' worth of cases as input and predict the number of new cases for the next time step. The number of LSTM layers was set to 64, there is no activation function since usually reggression dont need activation and dense was set to 1.
5) Model Training: Train the LSTM model using the prepared training dataset. This involves feeding the input sequences of the past 30 days' cases and training the model to predict the next day's number of new cases. Configure the training parameters by setting the number of epochs to 100.
6) Model Evaluation: Evaluate the trained LSTM model using the testing dataset. Measure the performance of the model using  mean absolute error (MAE). Assess the model's ability to accurately predict the number of new cases in Malaysia.
7) Model Deployment: Once satisfied with the model's performance, deploy it to make predictions on new, unseen data. Implement a mechanism to input the latest available data and obtain the model's predictions for future time steps.

## Results
The prediction results
-Graph Predicted VS Actual:

![Matplotlib Graph Predicted Vs Actual](farah-graph-predicted-vs-actual.png)

-Model Performance:

![Model Performance](farah-model-performance.png)

-Model Architecture:

![Model Architecture](farah-model-summary.png)

-Tensorboard MSE:

![Tensorboard MSE](farah-tensorboard-MSE.png)

## Credits
- https://github.com/MoH-Malaysia/covid19-public
