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

---

### 📁 `script` Folder
- **Purpose:** Contains all necessary scripts for:
  - Calculating returns.
  - Generating features for the model.

---

### 📄 `benchmark.ipynb`
- **Purpose:** Defines the benchmark and baseline strategies.
- **Usage:** Provides a comparative analysis to evaluate the model's performance.

---

## How to Run the Project
1. Navigate to the `data` folder and execute `data.ipynb` to prepare the datasets.
2. Use the scripts in the `script` folder to calculate returns and features.
3. Run `benchmark.ipynb` to compare model results with baseline strategies.

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