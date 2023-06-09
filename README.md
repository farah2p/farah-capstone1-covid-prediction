# COVID-19 Case Prediction using LSTM Neural Network
This project aims to create a deep learning model using LSTM neural network to predict new COVID-19 cases in Malaysia. The model utilizes the past 30 days of case data to forecast future case numbers. The goal is to achieve an MAPE (Mean Absolute Percentage Error) of less than 1% when tested against a testing dataset. The model's predictions will help determine if travel bans should be imposed or rescinded.
## Project Description
The year 2020 witnessed the global spread of the COVID-19 pandemic, leading to widespread disruptions and loss of lives. Scientists believe that the absence of AI-assisted automated tracking and predicting systems contributed to the rapid spread of the virus. Therefore, deep learning models, such as LSTM neural networks, can be utilized to predict daily COVID-19 cases accurately.
### Challenges Faced and Solutions
Challenges:
- Data collection: Ensuring the availability and reliability of daily COVID-19 case data for Malaysia.
- Model optimization: Determining the optimal number of LSTM nodes and the depth of the model to balance accuracy and efficiency.
- MAPE error control: Fine-tuning the model to achieve an MAPE error of less than 1% to ensure accurate predictions.

To overcome these challenges, we:
- Gathered the daily COVID-19 case data from https://github.com/MoH-Malaysia/covid19-public to ensure accurate and up-to-date information.
- Performed iterative experimentation and tuning to find the optimal architecture and hyperparameters for the LSTM model.
- Implemented regularization techniques and adjusted the model's parameters to minimize the MAPE error and improve prediction accuracy.
### Future Challenges and Features
Moving forward, we hope to address the following challenges and implement additional features:
- Incorporating external factors: Considering other variables such as vaccination rates, government interventions, and public sentiment to enhance the predictive capabilities of the model.
- Real-time prediction: Developing a system to provide real-time COVID-19 case predictions, allowing timely decision-making for public health authorities.

## Dataset
The dataset used in this project is sourced from the official data on the COVID-19 epidemic in Malaysia provided by MoH-Malaysia. The dataset includes historical daily case data, which will be used for training and testing the LSTM model.

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
## Model Criteria
- The model architecture should consist of LSTM, Dense, and Dropout layers.
- The LSTM layers should have a maximum of 64 nodes, and the model's depth can be adjusted based on requirements.
- The window size for input data should be set to 30 days.
- The Mean Absolute Percentage Error (MAPE) should be less than 1% when evaluated on the testing dataset. The MAPE is calculated using the formula:
  Mean Absolute Percentage Error = (Mean Absolute Error / (sum(actual)) * 100%
- The training loss displayed using TensorBoard.
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
## Project Output
Below are some sample visualizations generated by the project:
-Graph Predicted VS Actual:

![Matplotlib Graph Predicted Vs Actual](farah-graph-predicted-vs-actual.png)

-Model Performance:

![Model Performance](farah-model-performance.png)

-Model Architecture:

![Model Architecture](farah-model-summary.png)

-Tensorboard MSE:

![Tensorboard MSE](farah-tensorboard-MSE.png)

## Credits
The COVID-19 case data used in this project is sourced from:
https://github.com/MoH-Malaysia/covid19-public
## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.
