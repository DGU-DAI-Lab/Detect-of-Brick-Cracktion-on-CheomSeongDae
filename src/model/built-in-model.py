from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential([
    Conv2D(
        filters=48,
        kernel_size=(5,5),
        padding='same',
        activation='relu',
        input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(
        filters=96,
        kernel_size=(3,3),
        padding='same',
        activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.save('src/model/built-in-model.h5')