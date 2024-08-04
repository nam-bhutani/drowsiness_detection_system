import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from scipy.io import loadmat

# Load data from .mat files
data = loadmat("/xdata3.mat")
x_data = data['xdata']

data = loadmat("/ydata3.mat")
y_data = data['ydata']

# Convert y to categorical
label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(y_data)
y_data = tf.keras.utils.to_categorical(y_data, num_classes=2)

numFeatures = x_data.shape[1]
numHiddenUnits = 500 # accuracy
numClasses = 2
# Reshape x_data
x_data = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

#  the model definition
model = Sequential([
    LSTM(units=numHiddenUnits, input_shape=(1, numFeatures), return_sequences=False),
    Dense(units=numClasses, activation='softmax')
])

# Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=50, verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
acc = accuracy_score(y_test_classes, y_pred_classes)
print(f"Accuracy: {acc}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()
