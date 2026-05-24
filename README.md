# Interactive Data Cleaning & Preprocessing Tool

An interactive web application built using Python and Streamlit for performing common data cleaning and preprocessing tasks on CSV datasets.

## Features

- Upload CSV datasets
- Handle missing values
  - Drop rows
  - Fill with Mean
  - Fill with Median
  - Fill with Mode
- Automatic outlier detection using IQR
- Boxplot visualization for outlier inspection
- Categorical encoding
  - Label Encoding
  - One-Hot Encoding
- Feature scaling
  - Min-Max Scaling
  - Standardization
- Outlier handling
  - Remove outliers
  - Cap outliers
- Download cleaned dataset as CSV

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

```bash
git clone https://github.com/PrathviRaj60/Interactive-Data-Cleaning-and-Preprocessing-Tool.git
cd Interactive-Data-Cleaning-and-Preprocessing-Tool
```
Create virtual environment:
```bash
python -m venv venv
```
Activate virtual environment:
```bash
venv\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
streamlit run app.py
```
## Future Improvements:
- Correlation heatmaps
- Automated preprocessing recommendations
- Model training integration
- Advanced data profiling
- Preprocessing reports
---

Author
PrathviRaj
---
