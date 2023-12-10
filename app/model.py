import pickle5 as pickle
import tensorflow as tf
import tensorflow_hub as hub
from keras_preprocessing.image import ImageDataGenerator


def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        'media/dataset',
        target_size=(224, 224),
        class_mode='categorical')
    model_path = "D:/Models/tf2-preview_inception_v3_feature_vector_4"
    model = tf.keras.Sequential([
        hub.KerasLayer(hub.load(model_path), output_shape=[2048], trainable=False),
        tf.keras.layers.Dense(5, activation="sigmoid")
    ])
    model.build([None, 224, 224, 3])
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    H = model.fit(
        train_generator,
        epochs=5,
    )
    model.save_weights('media/model/model.h5')
    f = open('media/model/history.pckl', 'wb')
    pickle.dump(H.history, f)
    f.close()


if __name__ == "__main__":
    train_model()
