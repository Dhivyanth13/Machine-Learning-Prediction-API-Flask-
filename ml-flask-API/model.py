import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "scores": [20, 30, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["scores"]

model = LinearRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")