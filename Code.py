import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

df = pd.read_csv("iris.csv")

X = df.drop(columns=['petal_width', 'species_id'])
X = pd.get_dummies(X, columns=['species'], prefix_sep='=')
y = df['petal_width']

model = LinearRegression()
model.fit(X, y)

colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]

fig = px.bar(
    x=X.columns, y=model.coef_, color=colors,
    color_discrete_sequence=['red', 'blue'],
    labels=dict(x='Feature', y='Linear coefficient'),
    title='Weight of each feature for predicting petal width'
)
fig.show()
