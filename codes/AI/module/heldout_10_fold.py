import importlib.util
import os

def import_module_with_full_path(file_path):
    """
    Imports a Python module with its full path and assigns it to the specified aalias.

    Parameters:
        full_path (str): The full path of the module file.
        alias (str): The desired alias for the imported module.

    Returns:
        module: The imported module.
    """
    base_filename = os.path.basename(file_path)
    module_name = os.path.splitext(base_filename)[0]
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    imported_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(imported_module)
    return imported_module

def heldout_and_10_fold(data_file_path, metadata_file_path, scores_data_file_path, 
                        result_path, heldout_percentage = 0.15, model_name=None):    
    
    import pandas as pd
    import numpy as np
    import random

    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.random.set_seed(1015)
    
    DNN = import_module_with_full_path("/projects/unutmaz-lab/ruoyun/MECFS/all_tps/codes/AI/module/DNN.py")

    from sklearn.model_selection import StratifiedKFold
    import sklearn.metrics as sk_metrics
    
    from joblib import dump
    import pickle
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier

    glmnet = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=0.5, max_iter=100000000.0,
                    multi_class='auto', n_jobs=None, penalty='elasticnet',
                    random_state=1015, solver='saga', tol=0.0001, verbose=0,
                    warm_start=False)
    svm = SVC(C=2, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, probability=True, random_state=1015, shrinking=True,
      tol=0.0001, verbose=False)
    gdbt = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                             learning_rate=0.05, loss='deviance', max_depth=5,
                             max_features=30, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, 
                             min_samples_leaf=9, min_samples_split=60,
                             min_weight_fraction_leaf=0.0, n_estimators=1000,
                             n_iter_no_change=None,
                             random_state=1015, subsample=1, tol=0.0001,
                             validation_fraction=0.1, verbose=0,
                             warm_start=False)
    def build_three_layer_model(input_dim, activation_hidden, kernel_regularizer, optimizer):
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        inputs_norm = tf.keras.layers.BatchNormalization()(inputs)
        hidden1 = tf.keras.layers.Dense(64, activation=activation_hidden, 
                                        kernel_regularizer=kernel_regularizer)(inputs_norm)
        hidden1 = tf.keras.layers.Dropout(0.5)(hidden1)
        hidden1 = tf.keras.layers.BatchNormalization()(hidden1)
    
        hidden2 = tf.keras.layers.Dense(32, activation=activation_hidden, 
                                        kernel_regularizer=kernel_regularizer)(hidden1)
        hidden2 = tf.keras.layers.Dropout(0.5)(hidden2)
        hidden2 = tf.keras.layers.BatchNormalization()(hidden2)
    
        hidden3 = tf.keras.layers.Dense(12, activation=activation_hidden, 
                                        kernel_regularizer=kernel_regularizer)(hidden1)
        hidden3 = tf.keras.layers.Dropout(0.5)(hidden3)
        hidden3 = tf.keras.layers.BatchNormalization()(hidden3)
    
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
    
        model = tf.keras.models.Model(inputs=inputs, outputs = outputs)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    data = pd.read_csv(data_file_path, index_col=0).transpose()
    metadata = pd.read_csv(metadata_file_path, index_col=0)
    scores_data = pd.read_csv(scores_data_file_path, index_col=0)
    
    # get overlapped samples
    sample_id = data.index.intersection(scores_data.index)
    
    # exclude the inconsistent samples
    data = data.loc[sample_id,:]
    metadata = metadata.loc[sample_id,:]
    scores_data = scores_data.loc[sample_id,:]

    # get the Patient/Control as y0
    y0_orig = metadata.loc[:,'study_ptorhc'].map({'Control': 0, 'MECFS': 1})

    print(data.shape, scores_data.shape, y0_orig.shape)

    # get patient id and control id for sample-imbalance
    patient_id = list(y0_orig[y0_orig == 1].index)
    control_id = list(y0_orig[y0_orig == 0].index)
    
    activation_hidden = 'relu'
    batch_size = 64
    epochs=500
    
    crossvali_result_metrics = pd.DataFrame(None, index = range(10),
                              columns=['biomapai_train_acc', 'biomapai_test_acc','dnn_acc', 
                                       'gdbt_acc', 'glmnet_acc','svm_acc'])
    random.seed()
    random_state = random.randint(0, 1000)
    random.seed(random_state)
    held_out_size = round(len(control_id) * heldout_percentage)
    patient_sampled = random.sample(patient_id, held_out_size)
    control_sampled = random.sample(control_id, held_out_size)
    heldout_data = data.loc[patient_sampled + control_sampled,:]
    train_data = data.loc[[i for i in data.index.to_list() if i not in heldout_data.index],:]
    
    print(heldout_data.shape, train_data.shape)        
    
    scores= scores_data.loc[train_data.index,:]
    scores_heldout= scores_data.loc[heldout_data.index,:]
    y0 = y0_orig[train_data.index]
    y0_heldout = y0_orig[heldout_data.index]
    
    X = train_data
    y = scores
    y0 = y0
    kfold = StratifiedKFold(n_splits=10, random_state = random_state, shuffle=True)
    y_pred_table = pd.DataFrame(None, index = y0.index, 
                              columns=['y_true', 'biomapai_y_pred','dnn_y_pred','gdbt_y_pred', 
                                      'glmnet_y_pred', 'svm_y_pred'])
    y_pred_table.loc[y0.index, 'y_true'] = y0
    score_pred_table = pd.DataFrame(None, index = y0.index, columns=scores.columns)
    mse_list = []
    step_now = 0
    tf.keras.backend.clear_session()
    
    for train_index, test_index in kfold.split(X, y0):
        tf.keras.backend.clear_session()
        x_train, x_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
        y0_train, y0_test = y0[train_index], y0[test_index]
            
        ## DNN_Score_Model
        omics_score_model, omics_score_model_history = DNN.OmicScoreModel(epochs=epochs, 
               optimizer=tf.keras.optimizers.Adam(0.0005), batch_size=batch_size, 
               kernel_regularizer=tf.keras.regularizers.L2(0.008), 
               model_name = model_name).train(x_train, y_train, x_test, y_test)
        mse = DNN.OmicScoreModel().score(omics_score_model, x_test, y_test)
        mse_list.append(mse)
        print(f"MSE_step_{step_now}_{mse}")
        score_y_model_orig = DNN.ScoreLayer().build_model(y_train, y0_train)
        score_y_model = DNN.WeightsAdjust(omics_score_model, x_train, y_train, 
                                              score_y_model_orig).adjust_score_weight()
        train_accuracy = DNN.ScoreYModel(omics_score_model, score_y_model).evaluate(x_train, y0_train)[1]
        test_accuracy = DNN.ScoreYModel(omics_score_model, score_y_model).evaluate(x_test, y0_test)[1]
        test_predict_score = DNN.ScoreYModel(omics_score_model, score_y_model).get_score(x_test)
        test_predict_y = DNN.ScoreYModel(omics_score_model, score_y_model).predict(x_test).flatten()
        tf.keras.backend.clear_session()
            
         ## DNN_Model
        input_dim = x_train.shape[1]
        kernel_regularizer = tf.keras.regularizers.L2(0.008)
        optimizer = tf.keras.optimizers.Adam(0.0005)
        DNN_model = build_three_layer_model(input_dim, activation_hidden, kernel_regularizer, optimizer)
        DNN_history = DNN_model.fit(x_train, y0_train, batch_size=batch_size, epochs=epochs)
        _, DNN_accuracy = DNN_history.model.evaluate(x_test, y0_test)
        DNN_predict_y = DNN_history.model.predict(x_test).flatten()
        tf.keras.backend.clear_session()
            
        ## GDBT
        gdbt = gdbt.fit(x_train, y0_train)
        gdbt_predict_y = gdbt.predict(x_test)
        gdbt_accuracy = gdbt.score(x_test, y0_test)
        
        ## glmnet
        glmnet = glmnet.fit(x_train, y0_train)
        glmnet_predict_y = glmnet.predict(x_test)
        glmnet_accuracy = glmnet.score(x_test, y0_test)
        
        ## SVM
        svm = svm.fit(x_train, y0_train)
        svm_predict_y = svm.predict(x_test)
        svm_accuracy = svm.score(x_test, y0_test) 
            
        y_pred_table_i = pd.DataFrame([test_predict_y, DNN_predict_y, gdbt_predict_y, glmnet_predict_y, svm_predict_y])
        y_pred_table_i = y_pred_table_i.transpose()
        y_pred_table_i.index = y0_test.index
        y_pred_table_i.columns = ['biomapai_y_pred','dnn_y_pred','gdbt_y_pred', 'glmnet_y_pred', 'svm_y_pred']
        y_pred_table.loc[y0_test.index, ['biomapai_y_pred','dnn_y_pred','gdbt_y_pred', 'glmnet_y_pred', 'svm_y_pred']] = y_pred_table_i
        score_pred_table.loc[y0_test.index, :] = test_predict_score
        crossvali_result_metrics.loc[step_now,:] = train_accuracy, test_accuracy, DNN_accuracy, gdbt_accuracy, glmnet_accuracy, svm_accuracy
            
        print(f"Metrics_step_{step_now}_{crossvali_result_metrics.loc[step_now,'biomapai_train_acc']}")
        print(f"step_now_is {step_now}")
        step_now += 1
        tf.keras.backend.clear_session()
        
   # Now validate in heldout data
    heldout_result_metrics = pd.Series(None, index = ['biomapai', 'dnn', 'gdbt', 'glmnet','svm'])
    heldout_y_pred = pd.DataFrame(None, index = y0_heldout.index, 
                              columns=['y_true', 'biomapai_y_pred','dnn_y_pred','gdbt_y_pred', 
                                      'glmnet_y_pred', 'svm_y_pred'])
    heldout_y_pred.loc[y0_heldout.index, 'y_true'] = y0_heldout
    
   ## DNN_Score_Model
    omics_score_model, omics_score_model_history = DNN.OmicScoreModel(epochs=epochs, 
       optimizer=tf.keras.optimizers.Adam(0.0005), batch_size=batch_size, 
       kernel_regularizer=tf.keras.regularizers.L2(0.008), 
       model_name = model_name).train(X, y)
    score_y_model_orig = DNN.ScoreLayer().build_model(y, y0)
    score_y_model = DNN.WeightsAdjust(omics_score_model, X, y, score_y_model_orig).adjust_score_weight()
    heldout_mse =  DNN.OmicScoreModel().score(omics_score_model, heldout_data, scores_heldout)
    biomapai_acc_heldout = DNN.ScoreYModel(omics_score_model, score_y_model).evaluate(heldout_data, y0_heldout)[1]
    heldout_result_metrics['biomapai'] = biomapai_acc_heldout
    heldout_score_pred = DNN.ScoreYModel(omics_score_model, score_y_model).get_score(heldout_data)
    heldout_score_pred = pd.DataFrame(heldout_score_pred, index = y0_heldout.index, columns=scores_heldout.columns)
    biomapai_heldout_y_pred = DNN.ScoreYModel(omics_score_model, score_y_model).predict(heldout_data).flatten()
    heldout_y_pred.loc[:,'biomapai_y_pred'] = biomapai_heldout_y_pred
    tf.keras.backend.clear_session()
            
    ## DNN_Model
    input_dim = X.shape[1]
    kernel_regularizer = tf.keras.regularizers.L2(0.008)
    optimizer = tf.keras.optimizers.Adam(0.0005)
    DNN_model = build_three_layer_model(input_dim, activation_hidden, kernel_regularizer, optimizer)
    DNN_history = DNN_model.fit(X, y0, batch_size=batch_size, epochs=epochs)
    _, DNN_accuracy = DNN_history.model.evaluate(heldout_data, y0_heldout)
    DNN_predict_y = DNN_history.model.predict(heldout_data).flatten()
    heldout_result_metrics['dnn'] = DNN_accuracy
    heldout_y_pred.loc[:,'dnn_y_pred'] = DNN_predict_y
    tf.keras.backend.clear_session()
            
    ## GDBT
    gdbt = gdbt.fit(X, y0)
    gdbt_predict_y = gdbt.predict(heldout_data)
    gdbt_accuracy = gdbt.score(heldout_data, y0_heldout)
    heldout_result_metrics['gdbt'] = gdbt_accuracy
    heldout_y_pred.loc[:,'gdbt_y_pred'] = gdbt_predict_y
        
    ## glmnet
    glmnet = glmnet.fit(X, y0)
    glmnet_predict_y = glmnet.predict(heldout_data)
    glmnet_accuracy = glmnet.score(heldout_data, y0_heldout)
    heldout_result_metrics['glmnet'] = glmnet_accuracy
    heldout_y_pred.loc[:,'glmnet_y_pred'] = glmnet_predict_y
        
    ## SVM
    svm = svm.fit(X, y0)
    svm_predict_y = svm.predict(heldout_data)
    svm_accuracy = svm.score(heldout_data, y0_heldout) 
    heldout_result_metrics['svm'] = glmnet_accuracy
    heldout_y_pred.loc[:,'svm_y_pred'] = glmnet_predict_y
        
    
    result_values = [random_state, y0.index.to_list(), y0_heldout.index.to_list(), 
                     mse_list, crossvali_result_metrics, y_pred_table, score_pred_table, 
                     heldout_mse, heldout_result_metrics, heldout_y_pred, heldout_score_pred]
    result_keys = ['random_state', 'train_sample_list', 'heldout_sample_list',
                   'mse_list','result_metrics', 'y_pred_table_list', 'score_pred_table_list',
                  'heldout_mse', 'heldout_result_metrics', 'heldout_y_pred', 'heldout_score_pred']
    result = dict(zip(result_keys, result_values))
    
    result_path = f"{result_path.replace('.joblib', '')}_{random_state}.joblib"
    
    # save the results
    dump(result, result_path)