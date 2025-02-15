# Malavika Lavidi - AI Data Analyst

Hello! I'm Malavika Lavidi, an aspiring **AI Data Analyst** with a passion for solving complex data problems using machine learning, statistical analysis, and data visualization. I aim to use my skills to extract valuable insights from data and contribute to data-driven decision-making.

---

## 🚀 About Me

I specialize in:
- **Data Analysis**
- **Machine Learning**
- **Data Visualization**
- **Predictive Modeling**
- **Statistical Analysis**

I enjoy analyzing complex datasets to derive actionable insights and am excited about leveraging AI technologies to solve real-world problems. I am a continuous learner and always excited to explore new tools and techniques to improve my analytical skills.

---

## 🔧 Skills

Here are some of the skills I have developed and worked with:

- **Programming Languages**: Python, R, SQL
- **Data Analysis**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn, TensorFlow, Keras
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Statistical Analysis**: Regression Analysis, Hypothesis Testing
- **Database Management**: MySQL, PostgreSQL
- **Big Data Tools**: Hadoop, Spark
- **Web Development**: HTML, CSS, JavaScript
- **Version Control**: Git, GitHub
- **Cloud Computing**: AWS, Azure

---

## 📂 Projects

Here are some of the projects I have worked on:

### 1. **AI Model for Predicting House Prices**
   - **Tech**: Python, Scikit-learn, Pandas, Matplotlib
   - **Description**: Developed a machine learning model to predict house prices based on historical data. The model uses various features such as square footage, number of bedrooms, and location to predict the price of a house.

```python
# Example code for House Price Prediction
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('house_data.csv')

# Feature and target variables
X = data[['sqft_living', 'bedrooms', 'bathrooms']]
y = data['price']

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict prices
predictions = model.predict(X)
