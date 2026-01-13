🛠️ Industrial Time-Series Anomaly Detection

This project focuses on detecting anomalies in industrial sensor time-series data using four different models (unsupervised and supervised).
The goal is to compare model behavior, stability, and fault detection capability on real multi-sensor data.
📁 Dataset Structure
The dataset contains time-series sensor readings collected from an industrial system.
Sensors (8 features)
Accelerometer1RMS
Accelerometer2RMS
Current
Pressure
Temperature
Thermocouple
Voltage
Volume Flow RateRMS

Folder layout
data/
├── anomaly-free/      → normal operating data (training)
├── valve1/            → fault type 1
├── valve2/            → fault type 2
└── other/             → mixed / unknown faults

anomaly-free data is used for training unsupervised models
valve1, valve2, other folders are used only for testing
This makes the setup realistic:
➡️ models never see fault data during training (except supervised LSTM).
🧠 Models Used
I implemented four different anomaly detection models to understand their strengths and limitations.

1️⃣ Unsupervised LSTM Autoencoder

Type: Unsupervised Deep Learning
Idea:
Train an LSTM autoencoder on normal data only
The model learns to reconstruct normal patterns
High reconstruction error = anomaly
Training data:
anomaly-free.csv only
Testing:
Valve1, Valve2, Other folders
Metrics
Mean reconstruction error
Max reconstruction error
Anomaly ratio (percentage of anomalous points)
Key behavior
Very stable
Flags almost all faulty data as anomalous
Less sensitive to fault severity differences


2️⃣ Supervised LSTM Classifier
Type: Supervised Deep Learning
Idea:
Convert time-series into sequences
Train LSTM to directly classify normal vs anomaly
Training data
Normal + faulty samples with labels
Metrics
Accuracy
Precision
Recall
F1-Score
Key behavior
High accuracy on known fault types
Requires labeled data
Best choice when labels are available


3️⃣ MSET (Multivariate State Estimation Technique)
Type: Classical Unsupervised Model
Idea:
Store a memory matrix of normal states
Estimate new samples using nearest neighbors
Large estimation error = anomaly
Training data
anomaly-free data only
Metrics
Mean error
Max error
Anomaly ratio
Key behavior
Sensitive to subtle deviations
Better fault severity differentiation
Less stable than LSTM


4️⃣ Isolation Forest
Type: Tree-based Unsupervised Model
Idea:
Randomly isolate points using decision trees
Anomalies require fewer splits
Training data
anomaly-free data only
Metrics
Anomaly ratio
Key behavior
Fast and simple
Good baseline model
Sensitive to parameter tuning


📊 Model Comparison
We compare models using:
Anomaly ratio across files
Mean / max error
Stability (standard deviation of anomaly ratio)
Observations
LSTM Unsupervised → most stable, but less granular
Supervised LSTM → best accuracy when labels exist
MSET → captures severity differences well
Isolation Forest → simple, fast, but less consistent


⏱️ Time-Series Aspect
Sequential sensor data
Temporal dependency captured using LSTM
Sliding windows used for supervised learning
This makes the project more complex and realistic than standard tabular ML tasks.


📂 Saved Outputs
Each model saves:
Metrics CSV files
Trained model (where applicable)
Scalers and pipelines (.pkl)
Example:
result_lstm_s/
result_lstm_us/
result_mset/
result_isolation_forest/

🧪 Why Multiple Models?
Using multiple models allows:
Robust comparison
Understanding model behavior under faults
Better industrial decision-making
No single model is “best” in all situations.

✅ Conclusion
This project demonstrates:
Real industrial anomaly detection
Time-series modeling
Supervised vs unsupervised learning
Classical vs deep learning approaches
Proper evaluation and comparison

It is suitable for:
Academic projects
Research demonstrations
Industrial ML portfolios
