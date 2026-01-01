# Stock Price Prediction: LSTM vs. RNN (GOOGL)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat&logo=keras)
![Finance](https://img.shields.io/badge/Finance-Stock%20Analysis-green)

## Project Overview
This project implements and compares two types of Recurrent Neural Networks (RNNs) for **Time Series Forecasting**:
1.  **Long Short-Term Memory (LSTM):** Designed to capture long-term dependencies and prevent the vanishing gradient problem.
2.  **Simple RNN:** A basic recurrent network that captures short-term temporal dependencies.

The goal is to predict the **Next Day Closing Price** of Google (GOOGL) stock based on historical data from 2010 to 2024.

## Model Architectures

### 1. Long Short-Term Memory (LSTM)
* **Architecture:** Uses memory cells with gates (input, output, forget) to retain information over long sequences.
* **Configuration:** 50 LSTM units followed by a Dense output layer.
* **Advantage:** Better at learning trends in volatile financial data over longer periods.



### 2. Simple RNN
* **Architecture:** Uses a simple feedback loop to pass information from one step to the next.
* **Configuration:** 50 SimpleRNN units followed by a Dense output layer.
* **Limitation:** Struggles with long-term dependencies due to the vanishing gradient problem.



[Image of recurrent neural network architecture]


## Results & Comparison
Both models were trained on the same dataset and evaluated using **Mean Squared Error (MSE)** loss on the test set.

| Model | Test Loss (MSE) | Predicted Price (Example) | Conclusion |
| :--- | :--- | :--- | :--- |
| **LSTM** | **0.000879** | **$138.23** | Superior performance; captured the trend accurately. |
| **Simple RNN** | 0.002384 | $126.54 | Higher error; struggled to capture the complex pattern. |

**Winner:** The LSTM model significantly outperformed the Simple RNN, demonstrating its superior ability to handle financial time series data.

*(Note: Visualization of the predicted vs. actual stock prices is available in the notebook.)*

## Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Retrieval:** `yfinance` API
* **Data Processing:** Pandas, NumPy, Scikit-learn (MinMaxScaler)
* **Visualization:** Matplotlib

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PyPro2024/Stock-Price-Prediction-LSTM-vs-RNN.git]
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib yfinance scikit-learn
    ```
3.  **Run the Notebook:**
    Open `LSTMvsRNN_GOOGL_StockAnalyis.ipynb` in Jupyter Notebook or Google Colab. The notebook will automatically download the latest stock data.

---
*If you find this project helpful for your financial analysis, feel free to ⭐ the repo!*
