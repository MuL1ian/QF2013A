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
### 📄 `main.ipynb`
- Contains full function definitions and trace outputs for each trading algorithm.

### 📄 `visualizes.ipynb`
- **Purpose:** Visualise our trading strategies and compare them to the baseline.
- **Usage:** Runs the same functions as main.ipynb, but presents only user‑friendly visualisations. All function implementations live in the `script` folder.

---

## How to Run the Project
1. Navigate to the `data` folder and execute `data.ipynb` to prepare the datasets and you will get the 'Trading_project_data_clear.csv'.

2. For detailed of how each algorithms works, please run the `main.ipynb` This ipynb contain all the detailed function definition and print information for trace.

3. OR run `visualizes.ipynb` for a streamlined, illustrative analysis. It imports the same functions as `main.ipynb` from `script` but focuses purely on plotting performance.

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
