import mlflow
import mlflow.tensorflow

def train_basic_model(model, train_ds, val_ds, epochs, experiment_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="LungDisease_CNN"):
        mlflow.tensorflow.autolog()
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    return history
