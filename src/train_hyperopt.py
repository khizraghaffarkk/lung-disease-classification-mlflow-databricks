from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import tensorflow as tf
import mlflow
from tensorflow.keras import layers, models

def objective(params, input_shape, num_classes, train_ds, val_ds):
    tf.keras.backend.clear_session()
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = models.Sequential([
            layers.Resizing(input_shape[0], input_shape[1]),
            layers.Rescaling(1.0/255),
            layers.Conv2D(params['conv_filters'], (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(params['conv_filters'], (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(params['dense_units'], activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        mlflow.log_metric("val_accuracy", val_acc)
        return {'loss': -val_acc, 'status': STATUS_OK}

def run_hyperopt(input_shape, num_classes, train_ds, val_ds, experiment_name):
    search_space = {
        'conv_filters': hp.choice('conv_filters', [32, 64, 128]),
        'dense_units': hp.choice('dense_units', [64, 128, 256]),
        'learning_rate': hp.uniform('learning_rate', 1e-4, 1e-2)
    }
    trials = Trials()
    mlflow.set_experiment(experiment_name)
    best_params = fmin(fn=lambda params: objective(params, input_shape, num_classes, train_ds, val_ds),
                       space=search_space,
                       algo=tpe.suggest,
                       max_evals=5,
                       trials=trials)
    return best_params
