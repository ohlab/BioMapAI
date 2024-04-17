import importlib.util
import os

def import_module_with_full_path(file_path):
    """
    Imports a Python module with its full path and assigns it to the specified alias.

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

def cross_validate_model(data_file_path, metadata_file_path, scores_data_file_path, result_path, model_name=None): 
    
    import pandas as pd
    import numpy as np
    import random

    import tensorflow as tf
    tf.random.set_seed(1015)
    
    DNN = import_module_with_full_path("/projects/ohlab/ruoyun/MECFS/all_tps/codes/AI/module/DNN.py")

    from sklearn.model_selection import StratifiedKFold
    import sklearn.metrics as sk_metrics
    
    from joblib import dump
    import pickle
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier

    lr = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
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

    print(len(patient_id), len(control_id))
    
    activation_hidden = 'relu'
    batch_size = 64
    epochs=500
    
    mse_list = []
    step_now = 0
    sample_times = 100
    sample_time_now = 0
    y_pred_table_list = []
    score_pred_table_list = []
    result_metrics = pd.DataFrame(None, index = range(sample_times * 5),
                              columns=['train_acc', 'test_acc', 'dnn_acc', 'gdbt_acc', 'lr_acc', 'svm_acc'])

    for i in range(sample_times):
        random.seed()
        random_state = random.randint(0, 10000)
        random.seed(random_state)
        patient_sampled = random.sample(patient_id, len(control_id))
        train_data = data.loc[patient_sampled + control_id,:]
        scores= scores_data.loc[train_data.index,:]
        y0 = y0_orig[train_data.index]
        X = train_data
        y = scores
        y0 = y0
        kfold = StratifiedKFold(n_splits=5, random_state = random_state, shuffle=True)
        y_pred_table = pd.DataFrame(None, index = y0.index, 
                                    columns=['y_true', 'DNN_Score_y_pred','DNN_y_pred','GDBT_y_pred', 
                                             'LR_y_pred', 'SVM_y_pred'])
        y_pred_table.loc[y0.index, 'y_true'] = y0
        score_pred_table = pd.DataFrame(None, index = y0.index, columns=scores.columns)
        
        for train_index, test_index in kfold.split(X, y0):
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
            #test_predict_y = (test_predict_y.flatten() > 0.5).astype(int)
            tf.keras.backend.clear_session()
            
            ## DNN_Model
            input_dim = x_train.shape[1]
            kernel_regularizer = tf.keras.regularizers.L2(0.008)
            optimizer = tf.keras.optimizers.Adam(0.0005)
            DNN_model = build_three_layer_model(input_dim, activation_hidden, kernel_regularizer, optimizer)
            DNN_history = DNN_model.fit(x_train, y0_train, batch_size=batch_size, epochs=epochs)
            _, DNN_accuracy = DNN_history.model.evaluate(x_test, y0_test)
            DNN_predict_y = DNN_history.model.predict(x_test).flatten()
            #DNN_predict_y = (DNN_predict_y.flatten() > 0.5).astype(int)
            tf.keras.backend.clear_session()
            
            ## GDBT
            gdbt = gdbt.fit(x_train, y0_train)
            gdbt_predict_y = gdbt.predict(x_test)
            gdbt_accuracy = gdbt.score(x_test, y0_test)
            ## logistic
            lr = lr.fit(x_train, y0_train)
            lr_predict_y = lr.predict(x_test)
            lr_accuracy = lr.score(x_test, y0_test)
            ## SVM
            svm = svm.fit(x_train, y0_train)
            svm_predict_y = svm.predict(x_test)
            svm_accuracy = svm.score(x_test, y0_test) 
            
            y_pred_table_i = pd.DataFrame([test_predict_y, DNN_predict_y, gdbt_predict_y, lr_predict_y, svm_predict_y])
            y_pred_table_i = y_pred_table_i.transpose()
            y_pred_table_i.index = y0_test.index
            y_pred_table_i.columns = ['DNN_Score_y_pred','DNN_y_pred','GDBT_y_pred','LR_y_pred', 'SVM_y_pred']
            y_pred_table.loc[y0_test.index, ['DNN_Score_y_pred','DNN_y_pred','GDBT_y_pred','LR_y_pred', 'SVM_y_pred']] = y_pred_table_i
            score_pred_table.loc[y0_test.index, :] = test_predict_score
            result_metrics.loc[step_now,:] = train_accuracy, test_accuracy, DNN_accuracy, gdbt_accuracy, lr_accuracy, svm_accuracy
            
            print(f"Metrics_step_{step_now}_{result_metrics.loc[step_now,'test_acc']}")
            print(f"step_now_is {step_now}")
            step_now+=1
            tf.keras.backend.clear_session()
            
        y_pred_table_list.append(y_pred_table)
        score_pred_table_list.append(score_pred_table)
        print(f"sample_time_now {sample_time_now}")
        sample_time_now+=1
        
    result_values = [mse_list, result_metrics, y_pred_table_list, score_pred_table_list]
    result_keys = ['mse_list','result_metrics', 'y_pred_table_list', 'score_pred_table_list']
    result = dict(zip(result_keys, result_values))

    # save the results
    dump(result, result_path)