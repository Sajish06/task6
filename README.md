# 🧠 K-Nearest Neighbors (KNN) Classification on Iris Dataset

## 🎯 Objectives
- Implement **K-Nearest Neighbors (KNN)** algorithm to classify the Iris flower dataset.  
- Perform **data preprocessing** including feature scaling and train-test split.  
- Use **cross-validation** to select the best value of K.  
- Evaluate the model using **accuracy**, **confusion matrix**, and **classification report**.  
- Visualize **decision boundaries** using **PCA** for 2D representation.

---

## 🧩 Steps Performed

1. **Import Required Libraries**  
   Loaded essential Python packages for data analysis, visualization, and machine learning.

2. **Load Dataset**  
   - Uploaded `Iris.csv` into Colab using the file picker,  
     or loaded the built-in Iris dataset from `sklearn.datasets`.

3. **Explore the Data (EDA)**  
   - Displayed dataset shape, summary statistics, and class distribution.  
   - Verified column names (handled case-sensitive labels like “Species”).

4. **Prepare Features and Labels**  
   - Separated input features (`X`) and target label (`y`).  
   - Split dataset into **training (80%)** and **testing (20%)** subsets using `train_test_split`.

5. **Feature Scaling**  
   - Applied **StandardScaler** to normalize numerical features, ensuring fair distance computation in KNN.

6. **Train and Evaluate KNN**  
   - Initialized `KNeighborsClassifier` with `k=5`.  
   - Predicted labels on the test data.  
   - Evaluated performance using **accuracy**, **confusion matrix**, and **classification report**.

7. **Hyperparameter Tuning (Best K Selection)**  
   - Used **5-fold cross-validation** to test multiple K values (1–29).  
   - Chose the K giving highest average accuracy.

8. **Final Model Evaluation**  
   - Re-trained KNN with best K value and computed final test metrics.

9. **Decision Boundary Visualization (PCA)**  
   - Reduced features to 2D using **Principal Component Analysis (PCA)**.  
   - Plotted the **decision boundary** and **class clusters** using `matplotlib`.

10. **Results Interpretation**  
    - Observed clear separation between species in PCA space.  
    - Reported model accuracy and insights from confusion matrix.

---

## 🧰 Libraries Used
- **pandas** – Data handling and preprocessing  
- **numpy** – Numerical computations  
- **matplotlib** & **seaborn** – Visualization  
- **scikit-learn (sklearn)** –  
  - `train_test_split` for data splitting  
  - `StandardScaler` for normalization  
  - `KNeighborsClassifier` for model training  
  - `cross_val_score` for cross-validation  
  - `confusion_matrix`, `classification_report`, `accuracy_score` for evaluation  
  - `PCA` for dimensionality reduction  

---

## ▶️ How to Run

1. **Open in Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/).  
   - Upload the provided `.ipynb` notebook and your `Iris.csv` file.

2. **Install & Import Libraries**
   ```python
   !pip install -q scikit-learn pandas matplotlib seaborn
