import numpy as np
import requests
import json
import tensorflow as tf

def predict_local(model, img, class_names):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    return class_names[pred_index], round(100 * np.max(preds[0]), 2)

def predict_databricks(model_url, token, img_array, class_names):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = json.dumps({"instances": img_array.tolist()})
    response = requests.post(model_url, data=data, headers=headers)
    pred_probs = response.json()['predictions'][0]
    pred_index = np.argmax(pred_probs)
    return class_names[pred_index]
