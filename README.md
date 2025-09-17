# 🧹 Interactive Data Cleaning & Preprocessing Tool

An easy-to-use **Streamlit web app** that helps you clean and preprocess datasets without writing a single line of code.  
Upload your CSV file, handle missing values, detect outliers, encode categorical variables, scale numeric data, and finally export the cleaned dataset.  

---

## ✨ Features

-  **Upload CSV** → Load your dataset instantly.  
-  **Handle Missing Values** → Drop rows, or fill using **Mean / Median / Mode**.  
-  **Outlier Detection** → Automatically suggests numeric columns with potential outliers + boxplot visualization.  
-  **Categorical Encoding** → Apply **Label Encoding** or **One-Hot Encoding**.  
-  **Normalization / Standardization** → Scale data using **Min-Max Scaling** or **Z-score Standardization**.  
-  **Outlier Handling** → Remove or cap outliers interactively.  
-  **Export** → Download the cleaned dataset as a CSV file.  

---

## 📦 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yPrathviRaj60/data-cleaning-tool.git data-cleaning-tool
   cd data-cleaning-tool

(Optional but recommended) Create a virtual environment:
python -m venv env
source env/bin/activate    # Mac/Linux
env\Scripts\activate       # Windows

Install dependencies:
pip install -r requirements.txt


## ⚙️ Tech Stack

Python 3.9+
Streamlit
Pandas
Matplotlib & Seaborn
Scikit-learn

## 📝 License

This project is licensed under the MIT License – feel free to use and improve it.