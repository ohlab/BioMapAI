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

def run_model(data_file_path, metadata_file_path, scores_data_file_path, score_y_model_file_path, 
              result_path, omics_score_model_path, history_path, adjusted_score_layer_path, model_name=None): 
    
    import pandas as pd
    import numpy as np
    import random

    import tensorflow as tf
    tf.random.set_seed(1015)
    
    DNN = import_module_with_full_path("/projects/ohlab/ruoyun/MECFS/all_tps/codes/AI/module/DNN.py")

    from sklearn.model_selection import StratifiedKFold
    import sklearn.metrics as sk_metrics

    from sklearn.ensemble import GradientBoostingClassifier
    
    from joblib import dump
    import pickle
    
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

    best_acc = 0
    best_train_index, best_test_index = None, None
    best_omics_score_model, best_omics_score_model_history = None, None
    adjusted_score_layer = None
    patient_sampled_list = []
    mse_list = []
    step_now = 0
    sample_times = 100
    sample_time_now = 0
    result_metrics = pd.DataFrame(None, index = range(sample_times * 5),
                              columns=['train_acc', 'test_acc','TN', 'FN', 'FP', 'TP',
                                'gdbt_train_acc', 'gdbt_test_acc', 'gdbt_TN', 'gdbt_FN', 'gdbt_FP', 'gdbt_TP'])

    for i in range(sample_times):
        random.seed()
        random_state = random.randint(0, 10000)
        random.seed(random_state)
        patient_sampled = random.sample(patient_id, len(control_id))
        patient_sampled_list.append(patient_sampled)
        train_data = data.loc[patient_sampled + control_id,:]
        scores= scores_data.loc[train_data.index,:]
        y0 = y0_orig[train_data.index]
        X = train_data
        y = scores
        y0 = y0
        kfold = StratifiedKFold(n_splits=5, random_state = random_state, shuffle=True)
        for train_index, test_index in kfold.split(X, y0):
            x_train, x_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
            y0_train, y0_test = y0[train_index], y0[test_index]
            omics_score_model, omics_score_model_history = DNN.OmicScoreModel(epochs=500, 
               optimizer=tf.keras.optimizers.Adam(0.0005), batch_size=64, 
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
            test_predict_y = DNN.ScoreYModel(omics_score_model, score_y_model).predict(x_test)
            test_predict_y = (test_predict_y.flatten() > 0.5).astype(int)
            confusion = sk_metrics.confusion_matrix(y0_test, test_predict_y)
            confusion_fraction = confusion / confusion.sum(axis=1)
            TN, FN, FP, TP = confusion_fraction.reshape(4,)
            result_metrics.loc[step_now,['train_acc', 
               'test_acc','TN', 'FN', 'FP', 'TP']] = train_accuracy, test_accuracy, TN, FN, FP, TP
            print(f"Metrics_step_{step_now}_{result_metrics.loc[step_now,['train_acc', 'test_acc','TN', 'FN', 'FP', 'TP']]}")
        
            gdbt.fit(x_train, y0_train)
            gdbt_predict_y = gdbt.predict(x_test)
            gdbt_train_accuracy = gdbt.score(x_train, y0_train)
            gdbt_accuracy = gdbt.score(x_test, y0_test)
            confusion = sk_metrics.confusion_matrix(y0_test, gdbt_predict_y)
            confusion_fraction = confusion / confusion.sum(axis=1)
            gdbt_TN, gdbt_FN, gdbt_FP, gdbt_TP = confusion_fraction.reshape(4,)
            result_metrics.loc[step_now,['gdbt_train_acc', 'gdbt_test_acc', 'gdbt_TN', 'gdbt_FN', 
             'gdbt_FP', 'gdbt_TP']] = gdbt_train_accuracy, gdbt_accuracy, gdbt_TN, gdbt_FN, gdbt_FP, gdbt_TP       
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_train_index, best_test_index = train_index, test_index
                best_omics_score_model, best_omics_score_model_history = omics_score_model, omics_score_model_history
                adjusted_score_layer = score_y_model
        
            print(f"step_now_is {step_now}")
            step_now+=1
            tf.keras.backend.clear_session()
            
        print(f"sample_time_now {sample_time_now}")
        sample_time_now+=1
        
    result_values = [mse_list, result_metrics, patient_sampled_list, best_acc, best_train_index, best_test_index]
    result_keys = ['mse_list','result_metrics', 'patient_sampled_list', 
                   'best_acc', 'best_train_index', 'best_test_index']
    result = dict(zip(result_keys, result_values))

    # save the results
    dump(result, result_path)
    best_omics_score_model.save(omics_score_model_path)
    adjusted_score_layer.save(adjusted_score_layer_path)
    with open(history_path, 'wb') as file:
        pickle.dump(best_omics_score_model_history.history, file)