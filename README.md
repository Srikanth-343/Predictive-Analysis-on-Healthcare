# Predictive-Analysis-on-Healthcare

# A Study on Unraveling Business Solutions: a Predictive Analytics Journey

## Objective
This project aims to utilize predictive analytics techniques to forecast liver disease in patients based on crucial health markers. By analyzing a dataset containing features such as liver enzymes, bilirubin levels, age, gender, etc., we aim to develop predictive models that can accurately identify individuals at risk of liver disease. The insights gained from this analysis can assist healthcare providers in early detection and intervention, ultimately improving patient outcomes and reducing healthcare costs.

## Project Structure
- **data**: Contains the dataset used in the project: `indian_liver_patient.csv`
- 
- **notebooks**: Jupyter notebook for predictive analysis: `Predictive_analysis_for_Healthcare.ipynb`
- **scripts**: Python scripts for utility functions, data preprocessing, and model training.
- **visualizations**: Visual representations of key findings, insights, and model performance.

## Installation
To replicate this project locally, follow these steps:

1. Open Jupyter Notebook: run the Predictive_analysis_for_Healthcare.ipynb jupyter notebook file

## Data Exploration
- Conducted exploratory data analysis (EDA) to understand the distribution and characteristics of health markers.
- Identified potential risk factors and correlations with liver disease outcomes.
- Visualized trends and patterns using histograms, scatter plots, and heatmaps.

- <img width="466" alt="image" src="https://github.com/Srikanth-343/Predictive-Analysis-on-Healthcare/assets/57741770/1d8ebc53-d867-43ad-9c5e-400c629054ae">
- The correlation matrix graph outwardly addresses the relationships between mathematical highlights in the liver patient dataset. Each square in the heatmap relates to the 
  relationship coefficient between two elements, with hotter varieties showing more grounded positive connections and cooler tones addressing negative connections. This 
  permits fast recognizable proof of expected examples or affiliations, directing resulting examination

## Data Preprocessing
- Cleaned and prepared raw data for analysis, addressing missing values and outliers.
- Standardized numerical features and encoded categorical variables as necessary.
- Applied feature scaling and normalization techniques to ensure model convergence.

  ### Scale the data
  scaler = StandardScaler()
  X_clustering_scaled = scaler.fit_transform(X_clustering)
  print(X_clustering_scaled)

## Model Development
- Explored various predictive models such as logistic regression, decision trees, and random forests.
- Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.
- Tuned hyperparameters and conducted feature selection to optimize model performance.

  ### Apply K-Means clustering
  kmeans = KMeans(n_clusters=2, random_state=42)
  your_dataframe['Cluster'] = kmeans.fit_predict(X_clustering_scaled)

  ![image](https://github.com/Srikanth-343/Predictive-Analysis-on-Healthcare/assets/57741770/fd8d613a-6ba8-47cf-8dfb-8ef8596f609a)

  
  ### Create a Random Forest classifier
  rf_classifier = RandomForestClassifier(random_state=42)
  # Train the model
  rf_classifier.fit(X_train, y_train)
  
  # Make predictions on the test set
  y_pred = rf_classifier.predict(X_test)


## Model Evaluation
- Assessed model performance on test data using cross-validation and/or holdout validation.
- Interpreted model results to identify important predictors and risk factors for liver disease.
- Conducted sensitivity analysis to evaluate model robustness and generalization.
- ### Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  classification_rep = classification_report(y_test, y_pred)

  print(f"Accuracy: {accuracy:.2f}")
  print("Confusion Matrix:\n", conf_matrix)
  print("Classification Report:\n", classification_rep)

  ![image](https://github.com/Srikanth-343/Predictive-Analysis-on-Healthcare/assets/57741770/57b02c5e-0644-4148-a8dd-5720974cb67d)


## Results and Recommendations

- Feature Exploration and Selection: Research include significance, utilizing progressed methods for a more profound comprehension of factors impacting liver sickness 
  expectation.
- Address Class Imbalance: Moderate the noticed irregularity in arrangement models, particularly by zeroing in on further developing review for patients without liver 
  sickness.
- Advanced Clustering Techniques: Investigate refined bunching techniques past K-Means to uncover more multifaceted examples inside the dataset.

## Conclusion
- This task effectively explored the domains of prescient investigation and machine learning to resolve the basic issue of liver illness forecast. Utilizing calculated relapse, Irregular Woods characterization, and K-Means clustering, different aspects of information investigation were investigated. The calculated relapse model showed moderate precision, succeeding in distinguishing patients with liver illness however confronting difficulties with misleading negatives. Irregular Woodland exhibited heartiness, giving nuanced experiences. K-Means clustering outwardly divulged likely examples inside mathematical elements. In spite of triumphs, continuous difficulties incorporate further developing grouping balance and diving further into highlight significance. This task highlights the complicated exchange between information, models, and true critical thinking, giving an establishment to additional investigation and refinement in medical care prescient examination.


