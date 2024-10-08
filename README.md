# Time Series Forecasting with xLSTM
This repository contains Python code for experimenting with different LSTM architectures (xLSTM, LSTM, sLSTM, mLSTM) for time series forecasting using various datasets.

PLEASE READ [THIS](https://github.com/saurav-dhait/Time-Series-Forecasting-with-xLSTM/issues/1#issuecomment-2296614984) BEFORE YOU CONTINUE FORWARD.
## Project Structure

- `main.py`: This is the main script that contains the code for predicting results with xLSTM, sLSTM, mLSTM and LSTM.
- `xLSTM.py`: This file contains code for xLSTM, sLSTM and mLSTM.


## Workflow

### 1. Training Models
- Loading and preprocessing the dataset.
- Defining and training multiple LSTM architectures (xLSTM, LSTM, sLSTM, mLSTM).
- Plotting training losses for each model.
### 2. Evaluating Models
- Generate predictions using trained models.
- Calculate Mean Absolute Error (MAE) for each model.
- Visualize predictions against actual values.

## Results

![image](https://github.com/saurav-dhait/Time-Series-Forecasting-with-xLSTM/blob/main/assets/results.jpg)
## Requirements

You can install the required packages using the following command:

```sh
pip install -r requirements.txt
```

## Running the code
- To run the project, execute the following command (make sure you are in the project directory):

```sh
python main.py
```
## Resources
This repository contains slightly modified version of the original code [Link](https://github.com/akaashdash/xlstm)

Useful xLSTM repository [Link](https://github.com/AI-Guru/xlstm-resources?tab=readme-ov-file)
## Acknowledgements
This project is inspired by various tutorials and resources available for xLSTM.
