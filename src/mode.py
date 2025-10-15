import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes, conv_filters=64, dense_units=64):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(input_shape[0], input_shape[1]),
        layers.Rescaling(1.0/255)
    ])
    
    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(conv_filters, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(conv_filters, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
