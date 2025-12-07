import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import os
import warnings

# Try to import imblearn, usually needs 'pip install imbalanced-learn'
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("⚠️ Error: 'imbalanced-learn' library missing.")
    print("Please run: pip install imbalanced-learn")
    exit()

# Suppress all future warnings for cleaner output
warnings.filterwarnings("ignore")

def generate_mock_data(num_rows=10000):
    """
    Generates a small mock dataset for demonstration if the real file is not found.
    Mimics the structure and imbalance of a real transaction dataset.
    """
    print("--- WARNING: 'fraudTrain.csv' not found. Generating mock data. ---")
    np.random.seed(42)
    
    data = pd.DataFrame({
        'trans_date_trans_time': pd.to_datetime(pd.date_range('2023-01-01', periods=num_rows, freq='10min')),
        'cc_num': [f'40000000000{i}' for i in range(num_rows)],
        'category': np.random.choice(['gas_transport', 'groceries', 'online_retail', 'entertainment'], num_rows, p=[0.4, 0.3, 0.1, 0.2]),
        'amt': np.random.lognormal(mean=2.5, sigma=1.0, size=num_rows).round(2),
        'city_pop': np.random.randint(5000, 1000000, num_rows),
        'is_fraud': np.random.choice([0, 1], num_rows, p=[0.998, 0.002]), # Highly imbalanced
        'age': np.random.randint(20, 70, num_rows),
        'gender': np.random.choice(['M', 'F'], num_rows),
        'city': np.random.choice(['Columbus', 'Cleveland', 'Cincinnati'], num_rows),
        'state': ['OH'] * num_rows,
        'lat': np.random.uniform(38, 42, num_rows),
        'long': np.random.uniform(-85, -80, num_rows),
        'Unnamed: 0': range(num_rows)
    })
    
    # Introduce patterns for fraud
    data.loc[data['is_fraud'] == 1, 'amt'] = data.loc[data['is_fraud'] == 1, 'amt'] * np.random.uniform(2, 5)
    data.loc[data['is_fraud'] == 1, 'category'] = np.random.choice(['online_retail', 'jewelry', 'gaming'], data['is_fraud'].sum())
    
    return data

def load_and_engineer_data():
    """Load data or generate mock data, then perform feature engineering."""
    file_path = 'fraudTrain.csv'
    if os.path.exists(file_path):
        # NOTE: Using nrows=10000 for quick testing. Remove this argument for full training.
        data = pd.read_csv(file_path, nrows=10000)
    else:
        data = generate_mock_data()
        
    print(f"Data Loaded/Generated. Shape: {data.shape}")
    print(f"Fraudulent Transactions (1): {data['is_fraud'].sum()} ({data['is_fraud'].mean() * 100:.4f}%)")

    # Convert transaction time and extract features
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['hour'] = data['trans_date_trans_time'].dt.hour
    data['day_of_week'] = data['trans_date_trans_time'].dt.dayofweek
    data['log_amt'] = np.log1p(data['amt']) # Used for visualization

    # Simple Feature Engineering: High-Risk Category Flag
    high_risk_categories = ['online_retail', 'jewelry', 'gaming', 'misc_net']
    data['is_high_risk_category'] = data['category'].apply(lambda x: 1 if x in high_risk_categories else 0)

    # Drop unnecessary columns
    columns_to_drop = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 
        'street', 'zip', 'job', 'dob', 'trans_num', 'unix_time', 
        'merch_lat', 'merch_long' 
    ]
    data = data.drop(columns=columns_to_drop, errors='ignore')

    return data

def build_and_train_model(data):
    """Builds the ML pipeline, trains the model, and performs evaluation."""
    
    # Define Target and Features
    X = data.drop(['is_fraud', 'log_amt'], axis=1) # Drop log_amt as it's only for visualization
    y = data['is_fraud']

    # Split data (Stratify ensures equal fraud ratio in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Identify column types dynamically
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # Define Preprocessing Transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Define the Model with Class Weighting
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced', # Crucial for imbalanced data
        n_jobs=-1
    )

    # Create Imb-learn Pipeline: Preprocessing -> SMOTE -> Classifier
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        # SMOTE oversamples the minority class (Fraud) in the training data
        ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
        ('classifier', rf_classifier)
    ])

    # Training
    print("\n[ML] Starting Model Training (with SMOTE oversampling)...")
    model_pipeline.fit(X_train, y_train)
    print("[ML] Model Training Complete.")

    # Evaluation
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("🔥 Model Evaluation on Test Set (After SMOTE & Training) 🔥")
    print("="*50)
    print("--- Classification Report (Focus on Recall & Precision) ---")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))
    
    # Key Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score (AUPRC): {auprc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n--- Confusion Matrix ---")
    print(f"True Positives (Fraud Caught): {tp}")
    print(f"False Negatives (Fraud Missed): {fn}") # This is the costliest error to minimize
    print(f"False Positives (Legitimate Flagged): {fp}")
    print(f"True Negatives (Legitimate Correctly Ignored): {tn}")
    print(f"Goal: Minimize FN, maximize TP.")
    
    return model_pipeline

def plot_visualizations(data):
    """Generates the four requested visualizations."""
    
    print("\n[Viz] Preparing Visualizations... Windows will open simultaneously.")

    # --- GRAPH 1: Target Imbalance Distribution (Pie Chart) ---
    plt.figure(figsize=(6, 6))
    counts = data['is_fraud'].value_counts()
    labels = ['Legitimate (0)', 'Fraud (1)']
    
    plt.pie(counts, labels=labels, autopct='%1.4f%%', startangle=90, 
            colors=['#007FFF', '#FF4500'], explode=[0, 0.2], shadow=True,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
    plt.title('1. Transaction Class Distribution (Imbalance)')
    print("   -> Graph 1 (Pie Chart) prepared.")

    # --- GRAPH 2: Transaction Amount Distribution (Box and Histogram) ---
    plt.figure(figsize=(16, 6))
    
    # Box Plot: Comparing Amount by Class
    plt.subplot(1, 2, 1)
    sns.boxplot(x='is_fraud', y='amt', data=data, palette=['#007FFF', '#FF4500'])
    plt.title('2A. Transaction Amount Distribution by Class (Raw)')
    # Limit Y-axis for better visibility of the majority of transactions
    plt.ylim(0, 500)

    # Histogram: Distribution of Log Amount
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x='log_amt', hue='is_fraud', kde=True, bins=50, palette=['#007FFF', '#FF4500'])
    plt.title('2B. Distribution of Log(Amount) by Class')
    
    plt.tight_layout()
    print("   -> Graph 2 (Distributions) prepared.")

    # --- GRAPH 3: Temporal Analysis (Fraud Frequency) ---
    
    # Aggregate fraud counts by hour and day of week
    fraud_by_hour = data.groupby('hour')['is_fraud'].sum().reset_index()
    fraud_by_day = data.groupby('day_of_week')['is_fraud'].sum().reset_index()
    
    plt.figure(figsize=(16, 6))
    
    # Plot 1: Fraudulent Transactions by Hour of Day
    plt.subplot(1, 2, 1)
    sns.lineplot(x='hour', y='is_fraud', data=fraud_by_hour, marker='o', color='#FF4500', linewidth=3)
    plt.title('3A. Fraud Frequency by Hour of Day (0=Midnight)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Fraudulent Transactions')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Fraudulent Transactions by Day of Week
    plt.subplot(1, 2, 2)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.barplot(x='day_of_week', y='is_fraud', data=fraud_by_day, palette='cividis')
    plt.title('3B. Fraud Frequency by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Fraudulent Transactions')
    plt.xticks(ticks=range(7), labels=day_labels)
    
    plt.tight_layout()
    print("   -> Graph 3 (Time Analysis) prepared.")

    # --- GRAPH 4: Correlation Heatmap ---
    
    # Select only numerical columns for correlation calculation
    numerical_data = data.select_dtypes(include=np.number).drop(columns=['log_amt', 'hour', 'day_of_week'], errors='ignore')
    
    # Calculate the correlation matrix
    correlation_matrix = numerical_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True,          
        fmt=".2f",          
        cmap='rocket_r',     # Reversed rocket color map
        linewidths=.5,       
        linecolor='white'
    )
    plt.title('4. Correlation Heatmap of Numerical Features')
    print("   -> Graph 4 (Heatmap) prepared.")
    
    print("[Viz] Opening all windows now. Close ALL windows to proceed to Model Training.")
    plt.show() 

def demo_prediction(model_pipeline):
    """
    Simulates a live transaction to show judges the model in action.
    """
    print("\n" + "="*50)
    print("--- 🕵️ JUDGE'S LIVE DEMO (Simulation) ---")
    print("="*50)
    print("Simulating a new, suspicious transaction input...")
    
    # Create a single fake transaction dataframe
    # We now include ALL columns that exist in the training data to prevent errors
    new_transaction = pd.DataFrame({
        'Unnamed: 0': [0],          # Dummy index
        'category': ['online_retail'], 
        'amt': [1500.00],           
        'gender': ['M'],            # Dummy gender
        'city': ['Columbus'],       # Dummy city
        'state': ['OH'],            # Dummy state
        'lat': [39.96],             # Dummy latitude
        'long': [-83.00],           # Dummy longitude
        'city_pop': [50000],        
        'age': [30],
        'hour': [3],                # 3 AM (Unusual time)
        'day_of_week': [1],         # Tuesday
        'is_high_risk_category': [1] # Manually setting our feature flag
    })
    
    print(f"\nTransaction Details:")
    print(f" - Amount: $1500.00")
    print(f" - Category: Online Retail")
    print(f" - Time: 3:00 AM")
    
    # Make Prediction
    try:
        prediction = model_pipeline.predict(new_transaction)[0]
        probability = model_pipeline.predict_proba(new_transaction)[0][1]
        
        print("\n--- MODEL RESULT ---")
        if prediction == 1:
            print(f"🚨 ALERT: FRAUD DETECTED 🚨")
        else:
            print(f"✅ Transaction Approved (Legitimate)")
            
        print(f"Risk Score (Probability): {probability * 100:.2f}%")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[Demo Error] Could not run prediction: {e}")
        print("Ensure the input features match the training data exactly.")


if __name__ == '__main__':
    # 1. Load Data and Engineer Basic Features
    df = load_and_engineer_data()
    
    # 2. Run Visualizations
    plot_visualizations(df)
    
    # 3. Build and Train Machine Learning Model
    # We now capture the returned pipeline model
    trained_model = build_and_train_model(df)
    
    # 4. Run the Judge's Demo
    demo_prediction(trained_model)
