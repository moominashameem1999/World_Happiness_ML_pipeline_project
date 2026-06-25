import pandas as pd

#Task 10: Model Comparison & Conclusion
results_data = {
    "Model Type": [
        "Regression (Predicting Score)", 
        "Regression (Predicting Score)", 
        "Regression (Predicting Score)", 
        "Regression (Predicting Score)",
        "Classification (Predicting Label)", 
        "Classification (Predicting Label)", 
        "Classification (Predicting Label)",
        "Clustering (Unsupervised)"
    ],
    "Algorithm Name": [
        "Simple Linear Regression (SLR)", 
        "Multiple Linear Regression (MLR)", 
        "Polynomial Regression (Degree 2)", 
        "Ridge Regression (Alpha=0.1)",
        "Logistic Regression", 
        "Decision Tree (Depth=4)", 
        "Random Forest (100 Trees)",
        "K-Means Clustering (K=3)"
    ],
    "Primary Evaluation Metric": [
        "R² Score: 0.6419",   
        "R² Score: 0.8283",   
        "R² Score: 0.6517",   
        "R² Score: 0.8283",   
        "Accuracy: 90.63%",   
        "Accuracy: 90.63%",   
        "Accuracy: 90.63%",   
        "Silhouette Score: 0.3070" 
    ],
    "Error / Validation Metric": [
        "MSE: 0.5080",        
        "MSE: 0.2436",        
        "MAE: 0.5840 | RMSE: 0.7030",        
        "MSE: 0.2436 (at Best Alpha)",        
        "F1-Score (Happy): 0.90", 
        "F1-Score (Happy): 0.90", 
        "F1-Score (Happy): 0.90", 
        "Inertia Plateau: Observed at K=3"      
    ]
}

# 2. Convert into a polished DataFrame
df_summary = pd.DataFrame(results_data)

# 3. Adjust index layout to start from 1
df_summary.index = df_summary.index + 1

# 4. Print clean terminal layout
print("=======================================================================")
print("             WORLD HAPPINESS REPORT 2015 - FINAL METRICS               ")
print("=======================================================================")
print(df_summary.to_string())