import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

data = pd.read_csv("merc.csv")
new_data = data.sort_values("price", ascending=False).iloc[131:]
new_data = new_data[new_data.year != 1970]
new_data = new_data.drop("transmission", axis=1)
new_data = new_data.drop("model", axis=1)
new_data = new_data.drop("fuelType", axis=1)

y = new_data["price"].values
x = new_data.drop("price", axis=1).values

from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.fit_transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=X_train, y=Y_train, validation_data=(x_test, y_test), batch_size=50, epochs=300)

loss_data = pd.DataFrame(model.history.history)
loss_data.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error

predict_arr = model.predict(x_test)
print(mean_absolute_error(y_test, predict_arr))
