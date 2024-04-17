import numpy as np
import pandas as pd
np.random.seed(1015)

import tensorflow as tf
tf.random.set_seed(1015)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class ModelConstruct:
    def __init__(self, model_name=None, activation_hidden='relu', 
                 kernel_regularizer=None, 
                 optimizer=tf.keras.optimizers.Adam(0.01)):
        self.model_name = model_name
        self.activation_hidden = activation_hidden
        self.kernel_regularizer = kernel_regularizer
        self.optimizer = optimizer

    def custom_loss_metrics_last_activation(self, y_type):
        if y_type == 'binary':
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
            last_activation = 'sigmoid'
        elif y_type == 'categorical':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
            last_activation = 'softmax'
        elif y_type == 'continuous':
            loss = tf.keras.losses.Huber()
            metrics = ['mse']
            last_activation = 'relu'
        else:
            raise ValueError("Invalid Y type")
        return loss, metrics, last_activation

    def y_list_out_metrics(self, y_type_list, y_dim_list, previous_layer, hidden_nodes=8):
        out_layer_lists = {}
        custom_loss = {}
        custom_metrics = {}

        n_y = len(y_type_list)
        n_y_2 = len(y_dim_list)
        assert(n_y == n_y_2)

        for i in range(n_y):
            loss, metrics, last_activation = self.custom_loss_metrics_last_activation(y_type_list[i])
            model_i = tf.keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer=self.kernel_regularizer)(previous_layer)
            model_i = tf.keras.layers.Dense(y_dim_list[i], activation=last_activation, name="out{}".format(i))(model_i)
            out_layer_lists["out{}".format(i)] = model_i
            custom_loss["out{}".format(i)] = loss
            custom_metrics["out{}".format(i)] = metrics
        return out_layer_lists, custom_loss, custom_metrics

    def build_model(self, input_dim, y_type_list, y_dim_list):
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        inputs_norm = tf.keras.layers.BatchNormalization()(inputs)

        hidden1 = tf.keras.layers.Dense(64, activation=self.activation_hidden, 
                                        kernel_regularizer=self.kernel_regularizer)(inputs_norm)
        hidden1 = tf.keras.layers.Dropout(0.5)(hidden1)
        hidden1 = tf.keras.layers.BatchNormalization()(hidden1)

        hidden2 = tf.keras.layers.Dense(32, activation=self.activation_hidden, 
                                        kernel_regularizer=self.kernel_regularizer)(hidden1)
        hidden2 = tf.keras.layers.Dropout(0.5)(hidden2)
        hidden2 = tf.keras.layers.BatchNormalization()(hidden2)

        out_layer_lists, custom_loss, custom_metrics = self.y_list_out_metrics(y_type_list, y_dim_list, hidden2)

        model = tf.keras.models.Model(inputs=inputs, outputs=list(out_layer_lists.values()), name=self.model_name)
        model.compile(optimizer=self.optimizer, loss=custom_loss, metrics=custom_metrics)
        return model

class YDataProcessor:
    def __init__(self):
        pass

    def y_type(self, y):
        y_length = len(y)
        unique_y_length = len(np.unique(y))
        if unique_y_length < 2:
            raise ValueError("Invalid Y. Y cannot be a consistent value.")
        elif unique_y_length == 2:
            return "binary"
        elif unique_y_length < 6:
            return "categorical"
        else:
            return "continuous"

    def y_dataframe_to_list_for_model(self, y):
        if not isinstance(y, np.ndarray): y = np.asarray(y)
        y_dim = y.shape[1]
        n_sample = y.shape[0]
        y_list = []
        for i in range(y_dim):
            y_i = y[:, i].reshape(n_sample, 1)
            y_i_cata = self.y_type(y_i)
            if y_i_cata == 'binary' or y_i_cata == 'continuous':
                y_list.append(y_i)
            elif y_i_cata == 'categorical':
                num_classes = len(np.unique(y_i))
                y_i_reshaped = tf.keras.utils.to_categorical(y_i, num_classes=num_classes)
                assert(y_i_reshaped.shape == (n_sample, num_classes))
                y_list.append(y_i_reshaped)
            else:
                raise ValueError("Invalid Y")
        return y_list

    def y_trans(self, y):
        if not isinstance(y, np.ndarray): y = np.asarray(y)
        y_list = self.y_dataframe_to_list_for_model(y)
        y_type_list = [self.y_type(y[:, i]) for i in range(len(y_list))]
        y_dim_list = [i.shape[1] for i in y_list]
        return y_list, y_type_list, y_dim_list

class OmicScoreModel:
    def __init__(self, model_name=None, activation_hidden='relu', kernel_regularizer=None,
                 optimizer=tf.keras.optimizers.Adam(0.01), epochs=200, batch_size=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.y_data_processor = YDataProcessor()
        self.model_name = model_name
        self.activation_hidden = activation_hidden
        self.kernel_regularizer = kernel_regularizer
        self.optimizer = optimizer
        self.modelconstruct = ModelConstruct(model_name=self.model_name, 
                                             activation_hidden=self.activation_hidden, 
                                             kernel_regularizer=self.kernel_regularizer, 
                                             optimizer=self.optimizer)
        
    def to_array(self, data):
        if not isinstance(data, np.ndarray): 
            data = np.asarray(data)
        return data
    
    def y_preprocess_with_test(self, y_train, y_test):
        y_train = self.to_array(y_train)
        y_test = self.to_array(y_test)
        y_stack = np.vstack([y_train, y_test])
        n_y_train = range(0,y_train.shape[0])
        n_y_test = range(y_train.shape[0],y_stack.shape[0])
        y_list, y_type_list, y_dim_list = self.y_data_processor.y_trans(y_stack)
        y_train_list = [i[n_y_train,:] for i in y_list]
        y_test_list = [i[n_y_test,:] for i in y_list]
        return y_train_list, y_test_list, y_type_list, y_dim_list
        
    def y_preprocess_no_test(self, y):
        y = self.to_array(y)
        y_list, y_type_list, y_dim_list = self.y_data_processor.y_trans(y)
        return y_list, y_type_list, y_dim_list
    
    def train_no_test(self, x_train, y_train):
        x_train = self.to_array(x_train)
        y_train = self.to_array(y_train)
        y_list, y_type_list, y_dim_list = self.y_preprocess_no_test(y_train)
        x_train = StandardScaler().fit_transform(x_train)
        model = self.modelconstruct.build_model(input_dim = x_train.shape[1], 
                               y_type_list=y_type_list, y_dim_list=y_dim_list)
        if not self.batch_size: self.batch_size = x_train.shape[0]
        score_history = model.fit(x_train, y_list, epochs=self.epochs,
                                  batch_size=self.batch_size)
        self.model = model
        return self.model, score_history
    
    def train_with_test(self, x_train, y_train, x_test, y_test):
        x_train = self.to_array(x_train)
        y_train = self.to_array(y_train)
        x_test = self.to_array(x_test)
        y_test = self.to_array(y_test)
        y_train_list, y_test_list, y_type_list, y_dim_list = self.y_preprocess_with_test(y_train, y_test)
        
        x_train = StandardScaler().fit_transform(x_train)
       
        model = self.modelconstruct.build_model(input_dim = x_train.shape[1], 
                               y_type_list=y_type_list, y_dim_list=y_dim_list)
        if not self.batch_size: self.batch_size = x_train.shape[0]
        score_history = model.fit(x_train, y_train_list, epochs=self.epochs,
                                  batch_size=self.batch_size, 
                                  validation_data=(x_test, y_test_list))
        self.model = model
        return self.model, score_history
    
    def train(self, x_train, y_train, x_test=None, y_test=None):
        if x_test is not None and y_test is not None:
            return self.train_with_test(x_train, y_train, x_test, y_test)
        elif x_test is None and y_test is None:
            return self.train_no_test(x_train, y_train)
        else: raise ValueError("Please input X test and Y test!")

    def predict(self, model, x_test):
        x_test = self.to_array(x_test)
        x_test = StandardScaler().fit_transform(x_test)
        predicted_y = model.predict(x_test)
        predicted_y_reshaped = np.empty((x_test.shape[0], len(predicted_y)))
        for index, yi in enumerate(predicted_y):
            if yi.shape[1] > 1:
                yi = np.argmax(yi, axis=1)
            else: yi = yi.flatten()
            predicted_y_reshaped[:,index] = yi
        return predicted_y_reshaped
    
    def mse_list(self, y_true_table, y_pred_table):
        mse_list = []
        y_true = y_true_table
        y_pred = y_pred_table
        for i in range(y_true.shape[1]):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            if self.y_data_processor.y_type(y_true_i) == 'categorical':
                y_max = max(y_true_i)
                if y_max == 0:
                    y_max = y_max + 1e-8
                y_true_i = y_true_i / y_max
                y_pred_i = y_pred_i / y_max
            mse_i = mean_squared_error(y_true=y_true_i, y_pred=y_pred_i)
            mse_list.append(mse_i)
        return mse_list
    
    def score(self, model, x_test, y_test):
        x_test = self.to_array(x_test)
        y_test = self.to_array(y_test)
        predicted_scores = self.predict(model, x_test)
        mse = self.mse_list(y_test, predicted_scores)
        return mse

class ScoreYModel:
    def __init__(self, omics_model, pretrained_last_layer):
        self.OmicScoreModel = OmicScoreModel()
        self.omics_model = omics_model
        self.pretrained_last_layer = pretrained_last_layer
        
    def get_score(self, X):
        predicted_scores = self.OmicScoreModel.predict(self.omics_model, X)
        return predicted_scores
    
    def predict(self, X):
        predicted_scores = self.get_score(X)
        return self.pretrained_last_layer.predict(predicted_scores)
        
    def evaluate(self, X, y):
        predicted_scores = self.predict(X)
        return self.pretrained_last_layer.evaluate(predicted_scores, y)

class ScoreLayer:
    def __init__(self, optimizer=tf.keras.optimizers.legacy.Adam(0.01), 
                 kernel_regularizer=tf.keras.regularizers.l2(0.0008),
                 epochs=200):
        self.kernel_regularizer = kernel_regularizer
        self.optimizer = optimizer
        self.epochs = epochs

    def build_model(self, X, y):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Normalization(axis=-1))
        model.add(tf.keras.layers.Dense(1, input_dim=X.shape[1], activation='sigmoid', 
                                        kernel_regularizer=self.kernel_regularizer))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        model.fit(X, y, epochs=self.epochs)
        return model

class WeightsAdjust:
    def __init__(self, omics_model, omics_data, true_score, pretrained_score_layer):
        self.omics_model = omics_model
        self.omics_data = omics_data
        self.true_score = true_score
        self.pretrained_score_layer = pretrained_score_layer
        self.OmicScoreModel = OmicScoreModel()
    
    def get_adjust_weight(self):
        mse = self.OmicScoreModel.score(self.omics_model, self.omics_data, self.true_score)
        weights = 1 - np.array(mse)
        return weights
    
    def adjust_layer_weight(self, model, layer_num, weights):
        layer_weights = model.layers[layer_num].get_weights()
        old_weights = layer_weights[0]
        old_weights_flat = old_weights.flatten()
        old_weights_sum = sum(old_weights_flat)
        new_weights = old_weights_flat * weights
        new_weights_sum = sum(new_weights)
        frac = old_weights_sum / (new_weights_sum + 1e-8)
        new_weights = new_weights * frac
        new_weights = new_weights.reshape(old_weights.shape)
        layer_weights[0] = new_weights
        model.layers[layer_num].set_weights(layer_weights)
        return model
    
    def adjust_score_weight(self):
        weights = self.get_adjust_weight()
        model = self.pretrained_score_layer
        layer_num = 1
        return self.adjust_layer_weight(model, layer_num, weights)
    