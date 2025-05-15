ANN Regression Model - Estimated Salary Prediction

This project demonstrates how to build an **Artificial Neural Network (ANN)** using TensorFlow and Keras to predict the `EstimatedSalary` of bank customers based on various features from the `Churn_Modelling.csv` dataset.

📁 Dataset

The dataset used is `Churn_Modelling.csv`, containing 10,000 records with the following columns:

- CustomerID, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited

We drop irrelevant columns and apply preprocessing to convert categorical values into numeric format.



⚙️ Workflow

1. **Data Preprocessing**
- Drop columns: `RowNumber`, `CustomerId`, `Surname`
- Encode categorical features:
  - `Gender` → Label Encoding
  - `Geography` → One-Hot Encoding
- Scale numerical features using `StandardScaler`

2. **Model Architecture**
A simple feed-forward neural network with the following layers:
- Input layer
- Dense layer with 64 units and ReLU activation
- Dense layer with 32 units and ReLU activation
- Output layer with 1 unit (no activation, regression output)

3. **Model Compilation**
- Loss: `Mean Absolute Error (MAE)`
- Optimizer: `Adam`

 4. **Model Training**
- Split: 80% Training, 20% Testing
- Callbacks:
  - `EarlyStopping` (to avoid overfitting)
  - `TensorBoard` (for visualization)

📦 File Structure
├── churn_model_regression.py # Python code for data processing and training
├── Churn_Modelling.csv # Dataset file
├── label_encoder_gender.pkl # Saved LabelEncoder for 'Gender'
├── onehot_encoder_geo.pkl # Saved OneHotEncoder for 'Geography'
├── scaler.pkl # Saved StandardScaler
├── regressionlogs/ # TensorBoard logs
└── README.md # Project documentation


📊 TensorBoard

To visualize training metrics:
bash
tensorboard --logdir=regressionlogs/

🧠 Model Performance
The model was trained for up to 100 epochs with early stopping and achieved decreasing MAE values over time. Final validation MAE converged around ~50,000.

🧪 Requirements
Python 3.x
TensorFlow 2.x
Pandas
Scikit-learn
Matplotlib (optional)
Pickle

Install dependencies:
pip install -r requirements.txt

Create requirements.txt:
tensorflow
pandas
scikit-learn
matplotlib

📌 Future Improvements
Try more advanced models like XGBoost, or Random Forest for regression
Experiment with more hidden layers and dropout
Hyperparameter tuning
Predict other variables (e.g., Churn Exited) using classification ANN

📜 License
This project is open-source under the MIT License. Feel free to use and modify it for academic and learning purposes.
🙌 Author
Created by K.Prapulla Kumar . Feel free to connect!




