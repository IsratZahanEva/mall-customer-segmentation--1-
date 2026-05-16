# Mall Customer Segmentation

This project is a Streamlit web app for customer segmentation using K-Means clustering.

## Dataset

The app uses the `Mall_Customers.csv` dataset with these columns:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Features

- Dataset overview
- Summary statistics
- Gender distribution chart
- Elbow Method chart
- K-Means clustering with user-controlled K value
- Cluster visualization using Annual Income and Spending Score
- Cluster profiles table
- Download button for clustered dataset

## Project Structure

```text
mall-customer-segmentation/
├── app.py
├── requirements.txt
├── Mall_Customers.csv
└── README.md
```

## How to Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
or, python -m streamlit run app.py

```

## Deployment

1. Create a public GitHub repository.
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `Mall_Customers.csv`
   - `README.md`
3. Go to Streamlit Cloud.
4. Select your GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Click Deploy.

