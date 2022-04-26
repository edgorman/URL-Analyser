import os
import tensorflow as tf

from URLAnalyser.log import Log
from URLAnalyser.constants import MODEL_DATA_DIRECTORY


def load_model(filename: str) -> tf.keras.models.Model:
    return tf.keras.models.load_model(os.path.join(MODEL_DATA_DIRECTORY, filename))


def save_model(model: tf.keras.models.Model, filename: str) -> None:
    model.save(os.path.join(MODEL_DATA_DIRECTORY, filename))


def create_layers(input_dim: int, pen_layer: tuple, dropout_rate: int, ult_layer: tuple) -> tf.keras.models.Sequential:
    input = tf.keras.layers.Input(name='lexical_input', shape=(200,))
    embed = tf.keras.layers.Embedding(name='embedding', input_dim=input_dim, output_dim=32, input_length=200)(input)

    conv1 = tf.keras.layers.Conv1D(name='conv_filter_3', filters=256, kernel_size=(3,))(embed)
    conv1 = tf.keras.layers.MaxPooling1D(name='max_pool_3', pool_size=198)(conv1)
    conv1 = tf.keras.layers.Flatten(name='flatten_3')(conv1)

    conv2 = tf.keras.layers.Conv1D(name='conv_filter_4', filters=256, kernel_size=(4,))(embed)
    conv2 = tf.keras.layers.MaxPooling1D(name='max_pool_4', pool_size=197)(conv2)
    conv2 = tf.keras.layers.Flatten(name='flatten_4')(conv2)

    conv3 = tf.keras.layers.Conv1D(name='conv_filter_5', filters=256, kernel_size=(5,))(embed)
    conv3 = tf.keras.layers.MaxPooling1D(name='max_pool_5', pool_size=196)(conv3)
    conv3 = tf.keras.layers.Flatten(name='flatten_5')(conv3)

    dense = tf.keras.layers.concatenate([conv1, conv2, conv3], axis=1)
    dense = tf.keras.layers.Dense(pen_layer[0], name='connected_1', activation=pen_layer[1])(dense)
    dense = tf.keras.layers.Dropout(dropout_rate, name='dropout')(dense)
    dense = tf.keras.layers.Dense(ult_layer[0], name='connected_2', activation=ult_layer[1])(dense)

    final = tf.keras.layers.Dense(1, name='output', activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=input, outputs=final)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def get_input_dim(model: tf.keras.models.Sequential) -> int:
    try:
        return model.layers[1]._self_unconditional_dependency_names['embeddings'].shape[0]
    except Exception:
        Log.Error("Could not extract input dimension from embedding layer, check you are using the correct model.")
