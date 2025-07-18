# Engine Health Prognostic Analysis on NASA Turbofan Engine Dataset

A comprehensive analysis of turbofan engine health using statistical methods and machine learning to predict and prevent failures.

**Authors:** Shweta Tiwaskar, Aditya Suryawanshi, Aanish Deshmukh, Aditya Vyavhare

---

## Abstract

> This paper shows a thorough analysis of sensor readings from turbofan engines to predict health and improve maintenance. We utilize NASA CMAPSS data and subject it to numerous analysis techniques ranging from standard statistics to sophisticated regression to evaluate the deterioration of the engine. The approach begins with Hypothesis testing, including one-sample t-tests and chi-square tests, which confirms sensor reliability across operational cycles and validates parameter stability. We performed correlation analysis on the 21 sensor channels to find interdependencies and identify clusters of similar sensors. We used three regression models (Linear, Lasso, and Ridge) to forecast fan speed behavior, with the best model achieving an R² of 0.76. Finally, a PCA-Random Forest hybrid model was designed to project sensor features and estimate engine health on a 0–100 scale, successfully differentiating between healthy and failing engines. This research bridges the gap between traditional statistical process control and modern predictive maintenance in aerospace.

---

## Methodology

Our methodology transforms raw sensor data into an actionable health score through a five-step hybrid pipeline that guarantees both operational ease and predictive performance.

1.  **Data Preprocessing**: Raw NASA CMAPSS turbofan data is cleaned, smoothed using a rolling median filter, and scaled.
2.  **Hypothesis Testing**: Classical one-sample t-tests and chi-square tests are used to validate sensor stability and reliability over the engine's life, ensuring only statistically sound inputs are used.
3.  **Regression Analysis**: Lightweight regression models (Linear, Lasso, Ridge) are trained to predict critical parameters like fan speed, benchmarking their performance to find the most accurate model.
4.  **Dimensionality Reduction**: Principal Component Analysis (PCA) is used to reduce the validated sensor data into a lower-dimensional space, retaining 95% of the variance to improve efficiency.
5.  **Health Score Prediction**: The PCA scores are fed into a Random Forest Regressor, which outputs a continuous and interpretable health score from 0 (near failure) to 100 (healthy).

### Methodology Flow

Block diagram of Engine Health Prognostics Analysis <img width="865" height="663" alt="image" src="https://github.com/user-attachments/assets/9dd1d30d-2e44-4a29-8353-42965c89aae1" />

*Figure 1: Block diagram of the analysis methodology.*

---

## Experimental Setup and Results

The study was conducted on the NASA CMAPSS dataset, which includes data from 100 engines, each with 21 sensor channels. All experiments used Python 3.8 in a Visual Studio Code environment with an 80/20 training-testing split.

### 1. Hypothesis Testing

To ensure data reliability, one-sample t-tests and chi-square tests were conducted to identify and exclude unstable sensors. Sensors with p-values < 0.05 were considered unstable and removed from the analysis.

<img width="759" height="422" alt="image" src="https://github.com/user-attachments/assets/ce795923-826a-4317-8563-db4d2060e1a7" />

*Figure 2: Mean Comparison for sensor_3.*

<img width="765" height="392" alt="image" src="https://github.com/user-attachments/assets/3aa01fcc-f82b-4f4f-a914-a0a4d984f765" />

*Figure 3: Variance Comparison for sensor_3.*

### 2. Correlation Analysis

A Pearson correlation matrix was calculated for the validated sensors to identify interdependencies and aid in feature selection.

<img width="1011" height="522" alt="image" src="https://github.com/user-attachments/assets/b5893829-197d-43bb-a7b7-b6dd1f2b2c1c" />

*Figure 4: Heatmap of the feature correlation matrix.*

**Top Feature Correlations:**
* `sensor_9` - `sensor_14`: 0.9632
* `sensor_11` - `sensor_12`: -0.8469
* `sensor_4` - `sensor_11`: 0.8301
* `sensor_8` - `sensor_13`: 0.8261
* `sensor_7` - `sensor_11`: -0.8228

### 3. Regression Model Comparison

Three regression models were trained to predict fan speed (sensor_8). Ridge and Linear Regression emerged as the top performers, explaining approximately 78% of the variance in fan speed.

| Technique         | MSE    | RMSE   | MAE    | SD     | R²      |
| ----------------- | ------ | ------ | ------ | ------ | ------- |
| Linear Regression | 0.2091 | 0.4573 | 0.3641 | 0.4572 | 0.7796  |
| Ridge Regression  | 0.3641 | 0.4573 | 0.3641 | 0.4572 | 0.7796  |
| Lasso Regression  | 0.9495 | 0.9744 | 0.7753 | 0.9741 | -0.0006 |
*Performance metrics from the regression analysis.*

<img width="980" height="563" alt="image" src="https://github.com/user-attachments/assets/aa7b5e78-1cb9-49b1-8e92-605a228212d4" />

*Figure 5: Bar chart comparing the performance of the three regression models.*

### 4. Engine Health Predictions

A hybrid PCA-Random Forest model was used to generate a continuous health score (0-100) for all 100 engines in the test set.

<img width="998" height="473" alt="image" src="https://github.com/user-attachments/assets/81da8559-4f54-4090-8a10-031bdfc136db" />

*Figure 7: Predicted health scores for 100 engines.*

**Health Score Summary:**
* **Total Engines Analyzed**: 100
* **Average Health Score**: 51.80
* **Standard Deviation**: 29.99
* **Minimum Health Score**: 0.00
* **Maximum Health Score**: 100.00

**Engines with Lowest Health Scores (Most Degraded):**
* Engine 91: 0.00
* Engine 22: 1.13
* Engine 73: 4.70
* Engine 62: 6.87
* Engine 16: 7.05

These results demonstrate the model's ability to successfully identify severely degraded engines.

---

## Conclusion and Future Scope

This research successfully establishes a robust methodology for turbofan engine health monitoring by combining statistical validation with machine learning models. The integrated approach enhances sensor data reliability, enables accurate parameter prediction, and provides a transparent evaluation of overall engine health.

Future work will focus on integrating real-time data streams and adaptive learning algorithms to further improve the accuracy and responsiveness of engine health prognostics.
