# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
df = pd.read_csv('winequality-red.csv', delimiter=';')

# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Prima di fare lo split, controllare se ci sono numeri NaN o colonne da Codificare

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
# fit_transform solo su quelle di train
X_train = sc.fit_transform(X_train)

# Apply the transform to the test set
# Solo transform su quella di test perché il FIT verrà fatto dopo nella prediction
X_test = sc.transform(X_test)

# Print the scaled training and test datasets
print("Scaled training set:\n", X_train)
print("Scaled test set:\n", X_test)
