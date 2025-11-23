import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv('house.csv')

df = df.dropna(subset=['size', 'total_sqft', 'bath', 'balcony', 'price'])

def extract_bedrooms(x):
    try:
        return int(str(x).split(' ')[0])
    except:
        return None

df['bedrooms'] = df['size'].apply(extract_bedrooms)
df = df.dropna(subset=['bedrooms'])

def convert_sqft(x):
    try:
        if '-' in str(x):
            a, b = str(x).split('-')
            return (float(a.strip()) + float(b.strip())) / 2
        else:
            return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])

X = df[['total_sqft', 'bath', 'balcony', 'bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
