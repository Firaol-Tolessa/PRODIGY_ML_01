
import pandas as pd
# Load the CSV file for analysis
property = pd.read_csv("train.csv")
property_test = pd.read_csv("test.csv")
property.head()

#Analyze the distribution and relationship among column attribute value using scatter diagram
import seaborn as sns
import matplotlib.pyplot as plt	
# sns.pairplot(vars=['LotArea','SalePrice','BsmtFullBath','BsmtHalfBath','FullBath'],data=property)
# plt.scatter(x='LotArea', y='SalePrice', data=property)
# plt.show()

# Generate heat map to find the correlating column attribute
# correlation_matrix = property.corr()
# plt.figure(figsize=(15, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='rocket_r', linewidths=0.5, fmt='.2f')

# Select the most relating features to the sale price
x_train = property[['LotArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','GarageArea','GrLivArea']]
y_train = property['SalePrice']

# X_train, X_test, y_train, y_test = train_test_split( x, y, train_size = 0.7, test_size = 0.3, random_state = 100 )

# Load the Test data
property_test = pd.read_csv("test.csv")
property.head()

x_test = property[['LotArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','GarageArea','GrLivArea']]

# from sklearn.preprocessing import MinMaxScaler
# scalar=MinMaxScaler()
# X_train=scalar.fit_transform(X_train)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
r_sq = model.score(x_train, y_train)


print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")


y_pred = model.predict(x_test)


# Write to a csv file
import pandas as pd 

df = pd.DataFrame(columns=['id', 'SalePrice'])
id= 1461
for i,sale in enumerate(y_pred):
	  df.loc[i] = [id,sale]
	  id += 1

df.to_csv('output.csv', index=False)


