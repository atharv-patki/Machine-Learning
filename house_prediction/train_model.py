import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
df=pd.read_csv("house_prices.csv")
x=df[["Area","Bedrooms"]]
y=df[["Price"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
joblib.dump(model,"house_price_model.pkl")
print("Model trained and saved Successfully!!")