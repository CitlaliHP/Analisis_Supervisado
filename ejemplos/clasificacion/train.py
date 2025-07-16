import numpy as nu
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    "tiempo_minutos" : [5,8,12,2,20,15,3],
    "paginas_visitadas": [3,5,7,1,10,8,2],
    "compro" : [0,0,1,0,1,1,0] 
}

df = pd.DataFrame(data)

X = df[["tiempo_minutos","paginas_visitadas"]]
y = df["compro"]

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(X_test)
print(y_pred)