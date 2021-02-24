import pandas as pd   
from sklearn.model_selection import train_test_split

def clean_columns():
    df = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data")
    columns_to_drop = ["ID", "Age", "Experience", "ZIP Code", "Family", "Education", "Securities Account", "Online", "CreditCard"]
    df = df.drop(columns=columns_to_drop, axis=1)
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"ccavg":"cc_avg"})
    return  df

def split_data(df, x_features):
    x = df[x_features]
    y = df["personal_loan"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test