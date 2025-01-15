import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tkinter as tk
from tkinter import messagebox

# Load and preprocess the dataset
file_path = r"C:\\CI\\heart_attack_prediction_dataset.xlsx"
print(f"Using file path: {file_path}")
data = pd.read_excel(file_path)

# Process "Blood Pressure" column to handle values like "158/88"
def process_blood_pressure(bp):
    try:
        systolic, diastolic = map(float, bp.split("/"))
        return (systolic + diastolic) / 2  # Average of systolic and diastolic
    except:
        return np.nan  # Handle invalid values

# Apply the processing to the "Blood Pressure" column
data["Blood Pressure"] = data["Blood Pressure"].apply(process_blood_pressure)

# Drop rows with missing or invalid values
data.dropna(inplace=True)

# Extract features and target
features = data[["Age", "Cholesterol", "Blood Pressure"]]

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Define fuzzy logic system with membership functions and rules
age = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "age")
cholesterol = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "cholesterol")
blood_pressure = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "blood_pressure")
risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), "risk")

# Define fuzzy membership functions for each variable
age["low"] = fuzz.trimf(age.universe, [0, 0, 0.3])
age["medium"] = fuzz.trimf(age.universe, [0.2, 0.5, 0.8])
age["high"] = fuzz.trimf(age.universe, [0.7, 1, 1])

cholesterol["low"] = fuzz.trimf(cholesterol.universe, [0, 0, 0.3])
cholesterol["medium"] = fuzz.trimf(cholesterol.universe, [0.2, 0.5, 0.8])
cholesterol["high"] = fuzz.trimf(cholesterol.universe, [0.7, 1, 1])

blood_pressure["low"] = fuzz.trimf(blood_pressure.universe, [0, 0, 0.3])
blood_pressure["medium"] = fuzz.trimf(blood_pressure.universe, [0.2, 0.5, 0.8])
blood_pressure["high"] = fuzz.trimf(blood_pressure.universe, [0.7, 1, 1])

risk["low"] = fuzz.trimf(risk.universe, [0, 0, 0.4])
risk["high"] = fuzz.trimf(risk.universe, [0.6, 1, 1])

# Define fuzzy rules
rule1 = ctrl.Rule(age["low"] & cholesterol["low"] & blood_pressure["low"], risk["low"])
rule2 = ctrl.Rule(
    age["high"] | cholesterol["high"] | blood_pressure["high"], risk["high"]
)

# Create control system and simulation
risk_ctrl = ctrl.ControlSystem([rule1, rule2])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)


# Function to predict heart attack risk
def predict_heart_attack_risk(age_input, cholesterol_input, systolic_input, diastolic_input):
    # Calculate average blood pressure from systolic and diastolic
    avg_bp = (systolic_input + diastolic_input) / 2

    # Create a DataFrame with valid feature names for normalization
    input_data = pd.DataFrame(
        [[age_input, cholesterol_input, avg_bp]],
        columns=["Age", "Cholesterol", "Blood Pressure"],
    )

    # Normalize inputs
    normalized_input = scaler.transform(input_data)
    age_norm = normalized_input[0][0]
    cholesterol_norm = normalized_input[0][1]
    bp_norm = normalized_input[0][2]

    # Set inputs for the fuzzy system
    risk_sim.input["age"] = age_norm
    risk_sim.input["cholesterol"] = cholesterol_norm
    risk_sim.input["blood_pressure"] = bp_norm

    # Compute the risk
    risk_sim.compute()
    risk_value = risk_sim.output["risk"]

    # Interpret the output
    if risk_value <= 0.5:
        return f"Low Risk (Risk Score: {risk_value:.2f})"
    else:
        return f"High Risk (Risk Score: {risk_value:.2f})"


# GUI Implementation
def compute_risk():
    try:
        # Get inputs from the user
        age_input = float(age_entry.get())
        cholesterol_input = float(cholesterol_entry.get())
        systolic_input = float(systolic_entry.get())
        diastolic_input = float(diastolic_entry.get())

        # Compute risk
        result = predict_heart_attack_risk(age_input, cholesterol_input, systolic_input, diastolic_input)

        # Display result
        messagebox.showinfo("Heart Attack Risk", result)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


# Create main application window
root = tk.Tk()
root.title("Heart Attack Risk Predictor")
root.geometry("400x350")

# GUI Labels and Input Fields
tk.Label(root, text="Age:").pack(pady=5)
age_entry = tk.Entry(root)
age_entry.pack(pady=5)

tk.Label(root, text="Cholesterol:").pack(pady=5)
cholesterol_entry = tk.Entry(root)
cholesterol_entry.pack(pady=5)

tk.Label(root, text="Systolic Blood Pressure:").pack(pady=5)
systolic_entry = tk.Entry(root)
systolic_entry.pack(pady=5)

tk.Label(root, text="Diastolic Blood Pressure:").pack(pady=5)
diastolic_entry = tk.Entry(root)
diastolic_entry.pack(pady=5)

# Compute Button
compute_button = tk.Button(root, text="Compute Risk", command=compute_risk)
compute_button.pack(pady=20)

# Run the application
root.mainloop()
