# QF2103 Assignment 1 - Group Project

## Overview
This project involves processing and analyzing trading data for model development and evaluation. It includes data preparation, feature calculation, and benchmarking strategies.

---

## Project Structure

### 📁 `data` Folder
- **Purpose:** Contains the original dataset and transformation processes for data cleaning.
- **How to Use:**  
  1. Run the `data.ipynb` notebook.  
  2. This will convert the `Trading_Project_Data.csv` file into `Trading_Project_Data_Cleaned.csv`
  3. The `Trading_Project_Data_Cleaned.csv` will be use for further project deployment

---

### 📁 `script` Folder
- **Purpose:** Contains all necessary scripts for:
  - Calculating returns.
  - Generating features for the model.
  - Stock Selection Algorithms.

---

### 📄 `visualizes.ipynb`
- **Purpose:** visualizes our main trading strategies and graph comparsion with the baseline.
- **Usage:** Provides a comparative analysis to evaluate the model's performance.

---

## How to Run the Project
1. Navigate to the `data` folder and execute `data.ipynb` to prepare the datasets and you will get the 'Trading_project_data_clear.csv'.
3. Run `visualizes.ipynb` to compare model results with baseline strategies.

---

## Requirements
- Python 3.9 
- Jupyter Notebook
- Required libraries (e.g., pandas, numpy, matplotlib)

## env

```
conda create -n QF2103 python=3.10 -y
pip install -r requirements.txt
```