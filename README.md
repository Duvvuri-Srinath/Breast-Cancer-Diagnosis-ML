# Breast Cancer Classification Project README

## Project Overview

The Breast Cancer Classification project is part of a hackathon and aims to classify breast tumors as either benign or malignant using machine learning techniques. This README provides an overview of the project and its key components.

Obtained accuracy ( on test data) = 95% accurate in classifying cancer as either benign or malignant

## Dependencies

Before running the code, ensure that you have the following Python dependencies installed:

- `numpy` for numerical operations
- `pandas` for data manipulation
- `matplotlib` for data visualization
- `scikit-learn` for machine learning tasks
- `tensorflow` and `keras` for building and training neural networks

You can install these dependencies using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```


## Data Collection and Processing

# Importing dependencies

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
```


# Load and preprocess the dataset

```bash
dataset = pd.read_csv('/content/clean_data.csv')
data_frame = pd.DataFrame(dataset)
data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^Unnamed')]
```

# Display basic dataset information

```bash
print(data_frame.shape)
print(data_frame.info())
print(data_frame.isnull().sum())
print(data_frame.describe())
```


# Visualize the distribution of the 'diagnosis' column

```bash
print(data_frame['diagnosis'].value_counts())
print(data_frame.groupby('diagnosis').mean())
```


# Split the data into features (X) and target (y)

```bash
x = data_frame.drop(columns='diagnosis', axis=1)
y = data_frame['diagnosis']
```

# Split the data into training and testing sets

```bash
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
```

# Standardize the feature data

```bash
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)
```


## Building the neural network model

```bash
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(31,)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='sigmoid')
])
```


# Compiling and fitting the model

```bash
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_std, y_train_encoded, validation_split=0.1, epochs=10)
```


# Plotting the training history

```bash
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='lower right')
```

# Evaluating the model

```bash
loss, accuracy = model.evaluate(x_test_std, y_test_encoded)
print(accuracy)
```


## Building the predictive system

# Sample input data for prediction

```bash
input_data = (13.61, 24.69, 87.76, 572.6, 0.09258, 0.07862, 0.05285, 0.03085, 0.1761, 0.0613, 0.231, 1.005, 1.752, 19.83, 0.004088, 0.01174, 0.01796, 0.00688, 0.01323, 0.001465, 16.89, 35.64, 113.2, 848.7, 0.1471, 0.2884, 0.3796, 0.1329, 0.347, 0.079)
```


# Convert input data to a numpy array and reshape it

```bash
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
```

# Standardize the input data

```bash
input_data_std = scaler.transform(input_data_reshaped)
```

# Make predictions

```bash
prediction = model.predict(input_data_std)
prediction_label = [np.argmax(prediction)]

if prediction_label[0] == 0:
    print("Benign Tumour")
else:
    print("Malignant Tumour")
```


## Running the code

To run the code:

1. Ensure you have the required dependencies installed as mentioned above.

2. Replace the data source with your dataset if necessary.

3. Execute the code in your preferred Python environment.

4. Input your data for prediction in the "Building the Predictive System" section.

5. The code will output the predicted class label for the input data.