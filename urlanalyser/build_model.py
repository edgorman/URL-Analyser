"""
    build_model.py - Edward Gorman - eg6g17@soton.ac.uk
"""
import os
import json
import joblib
from keras import Model
from keras.models import model_from_json
from keras.layers import *
import keras.backend as K
from keras.utils import plot_model
from src import *


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def load_model(model_name, model_filename, model_config):
    if model_name == 'cnn':
        # Load keras model
        model_filename = model_filename + ".h5"
        if os.path.exists(model_filename):
            model = model_from_json(json.dumps(model_config))
            model.load_weights(model_filename)
            return model
    else:
        # Load sklearn model
        model_filename = model_filename + ".pkl"
        if os.path.exists(model_filename):
            return joblib.load(model_filename)

    return None


def build_cnn(feat_name, params):
    input = Input(name='lexical_input', shape=(200,))

    # EMBEDDING LAYER (Char/Word/Label)
    embed = None
    if feat_name == '1':
        embed = Embedding(name='', input_dim=len(URLCHAR_VOCAB) + 2, output_dim=32,
                          embeddings_initializer='random_uniform', input_length=200)(input)
    elif feat_name == '2':
        embed = Embedding(name='', input_dim=len(URLWORD_VOCAB) + 2, output_dim=32,
                          embeddings_initializer='random_uniform', input_length=200)(input)
    elif feat_name == '3':
        embed = Embedding(name='', input_dim=len(URLLABL_VOCAB) + 2, output_dim=32,
                          embeddings_initializer='random_uniform', input_length=200)(input)

    # CONVOLUTION LAYERS
    c1 = Conv1D(name='kernel_3', filters=256, kernel_size=(3,))(embed)
    c1 = MaxPooling1D(name='maxpool_3', pool_size=(200-3+1))(c1)
    c1 = Flatten(name='flatten_3')(c1)
    c2 = Conv1D(name='kernel_4', filters=256, kernel_size=(4,))(embed)
    c2 = MaxPooling1D(name='maxpool_4', pool_size=(200-4+1))(c2)
    c2 = Flatten(name='flatten_4')(c2)
    c3 = Conv1D(name='kernel_5', filters=256, kernel_size=(5,))(embed)
    c3 = MaxPooling1D(name='maxpool_5', pool_size=(200-5+1))(c3)
    c3 = Flatten(name='flatten_5')(c3)
    c4 = Conv1D(name='kernel_6', filters=256, kernel_size=(6,))(embed)
    c4 = MaxPooling1D(name='maxpool_6', pool_size=(200-6+1))(c4)
    c4 = Flatten(name='flatten_6')(c4)

    # FINAL LAYERS
    m = concatenate([c1, c2, c3, c4], axis=-1)
    m = Dense(params['first_neuron'], name='fully_connected', activation=params['first_activation'])(m)
    m = Dropout(params['dropout'], name='drop_'+str(params['dropout']))(m)
    m = Dense(params['second_neuron'], name='fully_connected_2', activation=params['second_activation'])(m)

    # OUTPUT LAYER
    output = Dense(1, name='output_prediction', activation='sigmoid')(m)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[recall_m])
    plot_model(model, to_file='../models/cnn/urlnetmodel.png')

    return model


def refine_cnn(x_train, y_train, x_test, y_test, params):
    input = Input(name='Input', shape=(200, ))

    # EMBEDDING LAYER (Char/Word/Label)
    embed = Embedding(name='Embedding', input_dim=params['embedding_size'], output_dim=32, embeddings_initializer='random_uniform', input_length=200)(input)

    # CONVOLUTION LAYERS
    c1 = Conv1D(name='Conv_Filter_3', filters=256, kernel_size=(3,))(embed)
    c1 = MaxPooling1D(name='MaxPool_3', pool_size=(200-3+1))(c1)
    c1 = Flatten(name='Flatten_3')(c1)
    c2 = Conv1D(name='Conv_Filter_4', filters=256, kernel_size=(4,))(embed)
    c2 = MaxPooling1D(name='MaxPool_4', pool_size=(200-4+1))(c2)
    c2 = Flatten(name='Flatten_4')(c2)
    c3 = Conv1D(name='Conv_Filter_5', filters=256, kernel_size=(5,))(embed)
    c3 = MaxPooling1D(name='MaxPool_5', pool_size=(200-5+1))(c3)
    c3 = Flatten(name='Flatten_5')(c3)
    c4 = Conv1D(name='Conv_Filter_6', filters=256, kernel_size=(6,))(embed)
    c4 = MaxPooling1D(name='MaxPool_6', pool_size=(200-6+1))(c4)
    c4 = Flatten(name='Flatten_6')(c4)

    # FINAL LAYERS
    m = concatenate([c1, c2, c3, c4], axis=-1)
    m = Dense(params['first_neuron'], name='Fully_Connected', activation=params['first_activation'])(m)
    m = Dropout(params['dropout'], name='Dropout')(m)
    m = Dense(params['second_neuron'], name='Fully_Connected_2', activation=params['second_activation'])(m)

    # OUTPUT LAYER
    output = Dense(1, name='Output', activation='sigmoid')(m)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[recall_m])

    out = model.fit(x_train, y_train,
                    batch_size=50,
                    epochs=1,
                    validation_data=[x_test, y_test],
                    verbose=0)

    return out, model
