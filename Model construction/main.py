import os
import numpy as np
import pandas as pd
import warnings
from itertools import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr, ttest_ind, levene
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.linear_model import LassoCV, lasso_path, ElasticNetCV, enet_path
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, classification_report

from sklearn.calibration import calibration_curve
from matplotlib.pyplot import MultipleLocator

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

import decisionCurve as dc



def featrueSelect_TTest(data, colNames):
    data_1 = data[data['Y'].isin([1])]
    data_2 = data[data['Y'].isin([0])]
    counts = 0
    index = []
    index_pvlaue = []
    for colName in colNames[1:]:
        if levene(data_1[colName], data_2[colName])[1] > 0.05:
            if ttest_ind(data_1[colName], data_2[colName])[1] < 0.05:
                counts += 1
                index.append(colName)
                index_pvlaue.append(ttest_ind(data_1[colName], data_2[colName])[1])

        else:
            if ttest_ind(data_1[colName], data_2[colName], equal_var=False)[1] < 0.05:
                counts += 1
                index.append(colName)
                index_pvlaue.append(ttest_ind(data_1[colName], data_2[colName], equal_var=False)[1])
    dic = dict(zip(index, index_pvlaue))
    return index, dic


def plotMESEs(model):
    MSEs = model.mse_path_
    MSEs_mean = np.apply_along_axis(np.mean, 1, MSEs)
    MSEs_std = np.apply_along_axis(np.std, 1, MSEs)

    plt.figure()  # dpi =300
    plt.errorbar(model.alphas_, MSEs_mean  
                 , yerr=MSEs_std 
                 , fmt="o"  
                 , ms=3  
                 , mfc='r'  
                 , mec='r'  
                 , ecolor='silver' 
                 , elinewidth=2 
                 , capsize=4  
                 , capthick=1)  
    plt.semilogx()
    plt.axvline(model.alpha_, color='black', ls="--")
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    ax = plt.gca()
    y_major_locator = MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()


def plotLasso(X, y, model, la_en=True):
    eps = 1e-2  # the smaller it is the longer is the path
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'indianred'])
    if la_en:
        print("Computing regularization path using the lasso...")
        alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

        plt.figure()
        neg_log_alphas_lasso = -np.log10(alphas_lasso)
        for coef_l, c in zip(coefs_lasso, colors):
            l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
        plt.axvline(-np.log10(model.alpha_), linestyle='--', color='indianred', label='alpha: CV estimate')
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('Lasso Path')
        plt.axis('tight')
        plt.show()
    else:
        print("Computing regularization path using the elastic net...")
        alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=model.l1_ratio, fit_intercept=False)

        plt.figure()
        neg_log_alphas_ = -np.log10(alphas_enet)
        for coef_l, c in zip(alphas_enet, colors):
            l1 = plt.plot(neg_log_alphas_, coef_l, c=c)
        plt.axvline(-np.log10(model.alpha_), linestyle='--', color='indianred', label='alpha: CV estimate')
        plt.xlabel('-Log(alpha)')
        plt.ylabel('coefficients')
        plt.title('ElasticNet Path')
        plt.axis('tight')
        plt.show()



def plotLasso_(X, y, model, alphas):
    alphas = np.logspace(-3, 1, 200)  # the smaller it is the longer is the path
    coefs = model.path(X, y, alphas=alphas, fit_intercept=False)[1].T
    plt.figure()
    plt.semilogx(model.alphas_, coefs, '-')
    plt.axvline(model.alpha_, color='black', ls='--')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficients')
    plt.show()



def lassoFeatures(X, y, file_outpath):
    alphas = np.logspace(-3, 1, 200)
    clf = LassoCV(alphas=alphas, cv=10, max_iter=20000).fit(X, y)
    m_log_alpha = -np.log10(clf.alpha_)
    m_log_alphas = -np.log10(clf.alphas_)

    coef = pd.Series(clf.coef_, index=X.columns)

    feature_names = X.columns
    importance = np.abs(clf.coef_)
    idx_third = importance.argsort()[-2]
    threshold = importance[idx_third] + 0.01

    idx_features = (-importance).argsort()[:9]
    name_features = np.array(feature_names)[idx_features]

    if sum(coef != 0) != 0:
        selected_coef = coef[coef != 0]
        lasso_feature = selected_coef.index

        X_lasso_result = X[lasso_feature]
        X_lasso_result.index = X.index
        Xy_lasso_result = pd.concat([y, X_lasso_result], axis=1, join='inner')


        mp_coef = pd.concat([selected_coef.sort_values()])
        print('lasso: ' + str(len(mp_coef)))
        print("intercept：" + str(clf.intercept_))
        print(mp_coef)


        R_score_coef = X_lasso_result[mp_coef.index] * mp_coef.values
        R_score_coef['Rscore'] = R_score_coef.apply(lambda x: x.sum(), axis=1)
        for index, row in R_score_coef.iterrows():
            tep = row['Rscore'] + clf.intercept_
            R_score_coef.loc[index]['Rscore'] = tep


        X_lasso = pd.concat([Xy_lasso_result, R_score_coef['Rscore']], axis=1, join='inner')
        X_lasso.to_csv(os.path.join(file_outpath, 'lasso_Rscore.csv'))


        return m_log_alphas, clf, mp_coef, X_lasso
    else:
        print("Lasso Over! ")
        return m_log_alphas, clf, None, None



def elasticNetFeatures(X, y, file_outpath):
    clf = ElasticNetCV(alphas=[0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.], l1_ratio=[.01, .1, .5, 0.75, .99]).fit(X, y)

    m_log_alpha = -np.log10(clf.alpha_)
    m_log_alphas = -np.log10(clf.alphas_)

    coef = pd.Series(clf.coef_, index=X.columns)


    if sum(coef != 0) != 0:
        selected_coef = coef[coef != 0]
        lasso_feature = selected_coef.index

        X_lasso_result = X[lasso_feature]
        X_lasso_result.index = X.index
        Xy_lasso_result = pd.concat([y, X_lasso_result], axis=1, join='inner')

        mp_coef = pd.concat([selected_coef.sort_values()])
        print('elasticNet： ' + str(len(mp_coef)))
        print("intercept：" + str(clf.intercept_))
        print(mp_coef)


        R_score_coef = X_lasso_result[mp_coef.index] * mp_coef.values
        R_score_coef['Rscore'] = R_score_coef.apply(lambda x: x.sum(), axis=1)
        for index, row in R_score_coef.iterrows():
            tep = row['Rscore'] + clf.intercept_
            R_score_coef.loc[index]['Rscore'] = tep


        X_lasso = pd.concat([Xy_lasso_result, R_score_coef['Rscore']], axis=1, join='inner')
        X_lasso.to_csv(os.path.join(file_outpath, 'elasticNet_Rscore.csv'))


        return m_log_alphas, clf, mp_coef, X_lasso
    else:
        print("Lasso over！ ")
        return m_log_alphas, clf, None, None



def Tuning_XG(cv_params, other_params, X, y):
    model2 = XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model2,
                                 param_grid=cv_params,
                                 cv=5,
                                 n_jobs=4,
                                 scoring='roc_auc')
    optimized_GBM.fit(X, y)
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    print(optimized_GBM.estimator)

    return optimized_GBM.best_params_

def Tuning_RF(X, y):
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    tuned_parameter = [{'min_samples_leaf': [1, 2, 3, 4], 'n_estimators': [50, 100, 200]}]
    clf = GridSearchCV(estimator=rfc, param_grid=tuned_parameter, cv=5, n_jobs=1)

    clf.fit(X, y)
    evalute_result = clf.cv_results_['mean_test_score']
    print(clf.estimator)

    return clf.best_params_


def startTuning(X, y):

    cv_params = {'n_estimators': [700, 800, 900, 1000, 1100, 1200]}
    other_params = {'objective': 'binary:logistic', 'scale_pos_weight': 3, 'learning_rate': 0.1, 'n_estimators': 800,
                    'max_depth': 5, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                    'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    n_estimators = Tuning_XG(cv_params, other_params, X, y)

    cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': n_estimators['n_estimators'], 'max_depth': 5,
                    'min_child_weight': 1,
                    'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    max_depth = Tuning_XG(cv_params, other_params, X, y)

    cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': n_estimators['n_estimators'],
                    'max_depth': max_depth['max_depth'], 'min_child_weight': max_depth['min_child_weight'], 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    gamma = Tuning_XG(cv_params, other_params, X, y)

    cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    other_params = {'learning_rate': 0.1, 'n_estimators': n_estimators['n_estimators'],
                    'max_depth': max_depth['max_depth'], 'min_child_weight': max_depth['min_child_weight'], 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': gamma['gamma'], 'reg_alpha': 0, 'reg_lambda': 1}
    subsample = Tuning_XG(cv_params, other_params, X, y)

    cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    other_params = {'learning_rate': 0.1, 'n_estimators': n_estimators['n_estimators'],
                    'max_depth': max_depth['max_depth'], 'min_child_weight': max_depth['min_child_weight'], 'seed': 0,
                    'subsample': subsample['subsample'], 'colsample_bytree': subsample['colsample_bytree'],
                    'gamma': gamma['gamma'], 'reg_alpha': 0, 'reg_lambda': 1}
    reg_alpha = Tuning_XG(cv_params, other_params, X, y)

    cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    other_params = {'learning_rate': 0.1, 'n_estimators': n_estimators['n_estimators'],
                    'max_depth': max_depth['max_depth'], 'min_child_weight': max_depth['min_child_weight'], 'seed': 0,
                    'subsample': subsample['subsample'], 'colsample_bytree': subsample['colsample_bytree'],
                    'gamma': gamma['gamma'], 'reg_alpha': reg_alpha['reg_alpha'], 'reg_lambda': reg_alpha['reg_lambda']}
    learning_rate = Tuning_XG(cv_params, other_params, X, y)

    other_params = {'learning_rate': learning_rate['learning_rate'], 'n_estimators': n_estimators['n_estimators'],
                    'max_depth': max_depth['max_depth'], 'min_child_weight': max_depth['min_child_weight'], 'seed': 0,
                    'subsample': subsample['subsample'], 'colsample_bytree': subsample['colsample_bytree'],
                    'gamma': gamma['gamma'], 'reg_alpha': reg_alpha['reg_alpha'], 'reg_lambda': reg_alpha['reg_lambda']}
    return other_params


def machineLearning(X_train, y_train, X_test, y_test):

    print('X_train: ' + str(len(X_train)))
    print(y_train.value_counts())
    print('X_test: ' + str(len(X_test)))
    print(y_test.value_counts())

    classifier_tree = DecisionTreeClassifier(class_weight='balanced', splitter='random', random_state=0)

    classifier_RF = RandomForestClassifier(n_estimators=50, min_samples_leaf=4, random_state=0)

    classifier_svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, probability=True)
	
    classifier_lg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                       intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                       penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
                                       verbose=0, warm_start=False)

    classifier_XGB = XGBClassifier(scale_pos_weight=2,
                                   learning_rate=0.01,
                                   n_estimators=700,  
                                   max_depth=4,  
                                   min_child_weight=5,  
                                   gamma=0.2,  
                                   subsample=0.8,  
                                   colsample_bytree=0.8,  
                                   reg_alpha=1,
                                   reg_lambda=0.05,
                                   objective='binary:logistic')


    aucTrain = {'lg': None, 'svm': None, 'rf': None, 'XGB': None, 'tree': None}
    aucTest  = {'lg': None, 'svm': None, 'rf': None, 'XGB': None, 'tree': None}

    model_lg = classifier_lg.fit(X_train, y_train)
    y_pred_proba_lg = model_lg.predict_proba(X_test)
    auc_roc_lg = roc_auc_score(y_test, y_pred_proba_lg[:, 1])
    aucTest['lg'] = auc_roc_lg
    y_pred_proba_lg = model_lg.predict_proba(X_train)
    auc_roc_lg = roc_auc_score(y_train, y_pred_proba_lg[:, 1])
    aucTrain['lg'] = auc_roc_lg


    model_tree = classifier_tree.fit(X_train, y_train)
    y_pred_proba_tree = model_tree.predict_proba(X_test)
    auc_roc_tree = roc_auc_score(y_test, y_pred_proba_tree[:, 1])
    aucTest['tree'] = auc_roc_tree
    y_pred_proba_tree = model_tree.predict_proba(X_train)
    auc_roc_tree = roc_auc_score(y_train, y_pred_proba_tree[:, 1])
    aucTrain['tree'] = auc_roc_tree

    model_rf = classifier_RF.fit(X_train, y_train)
    y_pred_proba_rf = model_rf.predict_proba(X_test)
    auc_roc_rf = roc_auc_score(y_test, y_pred_proba_rf[:, 1])
    aucTest['rf'] = auc_roc_rf
    y_pred_proba_rf = model_rf.predict_proba(X_train)
    auc_roc_rf = roc_auc_score(y_train, y_pred_proba_rf[:, 1])
    aucTrain['rf'] = auc_roc_rf

    model_svm = classifier_svm.fit(X_train, y_train)
    y_pred_proba_svm = model_svm.predict_proba(X_test)
    auc_roc_svm = roc_auc_score(y_test, y_pred_proba_svm[:, 1])
    aucTest['svm'] = auc_roc_svm
    y_pred_proba_svm = model_svm.predict_proba(X_train)
    auc_roc_svm = roc_auc_score(y_train, y_pred_proba_svm[:, 1])
    aucTrain['svm'] = auc_roc_svm

    model_XGB = classifier_XGB.fit(X_train, y_train)
    y_pred_proba_XGB = model_XGB.predict_proba(X_test)
    auc_roc_XGB = roc_auc_score(y_test, y_pred_proba_XGB[:, 1])
    aucTest['XGB'] = auc_roc_XGB
    y_pred_proba_XGB = model_XGB.predict_proba(X_train)
    auc_roc_XGB = roc_auc_score(y_train, y_pred_proba_XGB[:, 1])
    aucTrain['XGB'] = auc_roc_XGB

    print('LR->SVM->RF->XG->Tree')
    print("-----Train_AUC-----")
    for key, value in aucTrain.items():
        print(f"{'%.5f'%value}")
    print("-----Test_AUC-----")
    for key, value in aucTest.items():
        print(f" {'%.5f'%value}")


    names = ['Logistic Regression',
             'Random Forest',
             'SVM',
             'DecisionTree',
             'XGB']

    sampling_methods = [model_lg,
                        model_rf,
                        model_svm,
                        model_tree,
                        model_XGB
                        ]

    colors = ['crimson',
              'orange',
              'gold',
              'mediumseagreen',
              'steelblue',
              'mediumpurple'
              ]

    roc_graph = multi_models_roc(names, sampling_methods, colors, X_test, y_test)
    # roc_graph.show()

    roc_graph = multi_models_roc(names, sampling_methods, colors, X_train, y_train)
    # roc_graph.show()

def MLSVM(X_train, y_train, X_test, y_test, claFlag = 1):
  
    print('X_train: ' + str(len(X_train)))
    print(y_train.value_counts())
    print('X_test: ' + str(len(X_test)))
    print(y_test.value_counts())

    if claFlag == 1:
        clf = SVC(kernel='rbf', random_state=rd, probability=True)
    elif claFlag == 2:
        clf = DecisionTreeClassifier(class_weight='balanced', splitter='random', random_state=0)
    elif claFlag == 3:
        clf = RandomForestClassifier(n_estimators=50, min_samples_leaf=4, random_state=0)
    elif claFlag == 4:
        clf = KNeighborsClassifier(n_neighbors=6, p=2, algorithm='brute')
    elif claFlag == 5:
        clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                       intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                       penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
                                       verbose=0, warm_start=False)


    model = clf.fit(X_train, y_train)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thres = roc_curve(y_train, y_pred_train)


    optimal = thres[np.argmax(tpr + (1 - fpr))]
    print('AUC->Acc->Sen->Spec->PPV->NPV')
    int_auc, int_acc, int_sen, int_spec, int_ppv, int_npv, int_cm  = GetMetrics(y_train, y_pred_train, optimal)
    print('trian_CM:\n '+str(int_cm)+ "AUC->Acc->Sen->Spec->PPV->NPV: {:.3f} ".format(int_auc)+  "{:.1f} ".format(int_acc*100)+ "{:.1f} ".format(int_sen*100)+  "{:.1f} ".format(int_spec*100)+  "{:.1f} ".format(int_ppv*100)+  "{:.1f}".format(int_npv*100))
    ext_auc, ext_acc, ext_sen, ext_spec, ext_ppv, ext_npv, ext_cm = GetMetrics(y_test, y_pred_test, optimal)
    print(' test_CM:\n' + str(ext_cm) + "AUC->Acc->Sen->Spec->PPV->NPV: {:.3f} ".format(ext_auc) +  "{:.1f} ".format(ext_acc*100) +  "{:.1f} ".format(ext_sen*100) +  "{:.1f} ".format(ext_spec*100)+  "{:.1f} ".format(ext_ppv*100)+ "{:.1f} ".format(ext_npv*100))


    yProbas = [y_pred_test,
               y_pred_train]
    yReals = [y_test, y_train]
    y_pro = pd.concat([pd.DataFrame(np.array(y_pred_test),columns =['y_pred_test']), pd.DataFrame(np.array(y_test),columns =['y_test']), pd.DataFrame(np.array(y_pred_train),columns =['y_pred_train']), pd.DataFrame(np.array(y_train),columns =['y_train'])], axis=1, join='inner')
    y_pro.to_csv('../data/selectedFeatures/1.csv')

    names = ['Testing cohort: DL+Radiomics',
             'Training cohort: DL+Radiomics']

    colors = ['crimson',
              'orange',
              'steelblue',
              'gold',
              'mediumseagreen',
              'mediumpurple'
              ]
    linestyles = ['--','-']

    roc_graph = multi_models_roc_(names, yProbas, colors, linestyles, yReals)
    roc_graph.show()

def multi_models_roc_(names, yProbas, colors, linestyles, yReals, save=False, dpin=100):


    plt.figure(figsize=(6, 5), dpi=dpin)

    for (name, yProba, colorname, line, yReal) in zip(names, yProbas, colors, linestyles, yReals):
        fpr, tpr, thresholds = roc_curve(yReal, yProba, pos_label=1)

        plt.plot(fpr, tpr, lw=1,linestyle =line, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')
        # plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=15)
        plt.legend(loc='lower right', fontsize=12)

    if save:
        plt.savefig('multi_models_roc.png')

    return plt


def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=False, dpin=100):

    plt.figure(figsize=(6, 6), dpi=dpin)

    for (name, method, colorname) in zip(names, sampling_methods, colors):
        y_test_predprob = method.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)


        plt.plot(fpr, tpr,lw=3, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=3, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=15)
        plt.legend(loc='lower right', fontsize=12)

    if save:
        plt.savefig('multi_models_roc.png')

    return plt



def plot_calibration_curve(names, yReals, yProbas, lines):
    plt.figure(figsize=(6, 4.5), dpi=100)
    index = 0
    for (name, yReal, yProba, line) in zip(names, yReals, yProbas, lines):
        if index == 2:
            break
        fraction_of_positives, mean_predicted_value = calibration_curve(yReal, yProba, n_bins=4)
        plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')
        plt.plot(mean_predicted_value, fraction_of_positives, color = 'teal', marker='.',linestyle=line,label='{}'.format(name))
        plt.xlabel("Predicted probability of LNM")
        plt.ylabel("Actual rate of LNM")
        plt.legend(loc='lower right', fontsize=12)
        #plt.title('calibration_curve')
        index = index +1

    plt.show()


def ROC_CaCurv(names, yProbas, colors, linestyles, yReals, save=False, dpin=100):
    fig, ax = plt.subplots(figsize=(20, 20), dpi=dpin)
    plt.figure(figsize=(20, 20), dpi=dpin)
    yProbas_ = [yProbas[0],
               yProbas[2]]
    names_ = ['ResNet',
             'Radiomics'
             ]
    yReals_ = [yReals[0],yReals[2]]
    plt.subplots(2, 2, 1)
    for (name, yProba, colorname,  yReal) in zip(names_, yProbas_, colors,  yReals_):
        fpr, tpr, thresholds = roc_curve(yReal, yProba, pos_label=1)
        plt.plot(fpr, tpr, lw=1, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')
        # plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('a', fontsize=12,x=-0.03,y=1.03)
        plt.legend(loc='lower right', fontsize=9)

    yProbas_ = [yProbas[1],
                yProbas[3]]
    names_ = ['ResNet',
              'Radiomics'
              ]
    yReals_ = [yReals[1], yReals[3]]
    plt.subplots(2, 2, 2)
    for (name, yProba, colorname, yReal) in zip(names_, yProbas_, colors, yReals_):
        fpr, tpr, thresholds = roc_curve(yReal, yProba, pos_label=1)
        plt.plot(fpr, tpr, lw=1, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('a', fontsize=12, x=-0.03, y=1.03)
        plt.legend(loc='lower right', fontsize=9)



    index = 0
    plt.subplots(2, 2, 3)
    names=['Testing cohort',
              'Training cohort'
           ]
    for (name, yReal, yProba, line) in zip(names, yReals, yProbas, linestyles):
        if index == 2:
            break
        fraction_of_positives, mean_predicted_value = calibration_curve(yReal, yProba, n_bins=4)
        plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')
        plt.plot(mean_predicted_value, fraction_of_positives, color='teal', marker='.', linestyle=line,
                 label='{}'.format(name))
        plt.xlabel("Predicted probability of LNM", fontsize=12)
        plt.ylabel("Actual rate of LNM", fontsize=12)
        plt.legend(loc='lower right', fontsize=9)
        plt.title('b', fontsize=12, x=-0.03,y=1.03)
        index = index + 1

    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = dc.calculate_net_benefit_model(thresh_group, yProbas[0], yReals[0])
    net_benefit_all = dc.calculate_net_benefit_all(thresh_group, y_test)
    net_benefit_model1 = dc.calculate_net_benefit_model(thresh_group, yProbas[2], yReals[2])
    net_benefit_all1 = dc.calculate_net_benefit_all(thresh_group, yReals[2])
    net_models = [net_benefit_model, net_benefit_model1]
    net_alls = [net_benefit_all, net_benefit_all1]
    nases = ['ResNet', 'Radiomics']

    fig, ax = plt.subplots(2, 2, 4)
    ax = dc.plot_DCAs(ax, thresh_group, net_models, net_alls, colors, nases)
    plt.show()


def ROC_CaCurv_(names, yProbas, colors, linestyles, yReals, save=False, dpin=100):
    fig, ax = plt.subplots(nrows=2,ncols=2)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

    # plt.figure(figsize=(20, 20), dpi=dpin)
    yProbas_ = [yProbas[0],
               yProbas[2]]
    names_ = ['ResNet',
             'Radiomics'
             ]
    yReals_ = [yReals[0],yReals[2]]
    # plt.subplots()
    for (name, yProba, colorname,  yReal) in zip(names_, yProbas_, colors,  yReals_):
        fpr, tpr, thresholds = roc_curve(yReal, yProba, pos_label=1)
        ax[0,0].plot(fpr, tpr, lw=1, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        ax[0,0].plot([0, 1], [0, 1], '--', lw=1, color='grey')
        # plt.axis('square')
        ax[0,0].set_xlim([0, 1])
        ax[0,0].set_ylim([0, 1])
        ax[0,0].set_xlabel('False Positive Rate', fontsize=12)
        ax[0,0].set_ylabel('True Positive Rate', fontsize=12)
        ax[0,0].set_title('A',  x=-0.03,y=1.03,fontdict= {'family':'Arial','weight': 'semibold', 'fontsize': 12})
        ax[0,0].legend(loc='lower right', fontsize=9)


    yProbas_ = [yProbas[1],
                yProbas[3]]
    names_ = ['ResNet',
              'Radiomics'
              ]
    yReals_ = [yReals[1], yReals[3]]
    for (name, yProba, colorname, yReal) in zip(names_, yProbas_, colors, yReals_):
        fpr, tpr, thresholds = roc_curve(yReal, yProba, pos_label=1)
        ax[0, 1].plot(fpr, tpr, lw=1, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        ax[0, 1].plot([0, 1], [0, 1], '--', lw=1, color='grey')
        # plt.axis('square')
        ax[0, 1].set_xlim([0, 1])
        ax[0, 1].set_ylim([0, 1])
        ax[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        ax[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        ax[0, 1].set_title('B', fontdict= {'family':'Arial','weight': 'semibold', 'fontsize': 12}, x=-0.03, y=1.03)
        ax[0, 1].legend(loc='lower right', fontsize=9)



    index = 0
    names=['Testing cohort',
              'Training cohort'
           ]
    for (name, yReal, yProba, line) in zip(names, yReals, yProbas, linestyles):
        if index == 2:
            break
        fraction_of_positives, mean_predicted_value = calibration_curve(yReal, yProba, n_bins=4)
        ax[1, 0].plot([0, 1], [0, 1], '--', lw=1, color='grey')
        ax[1, 0].plot(mean_predicted_value, fraction_of_positives, color='teal', marker='.', linestyle=line,
                 label='{}'.format(name))
        ax[1, 0].set_xlabel("ResNet for predicted probability of LNM", fontsize=12)
        ax[1, 0].set_ylabel("Actual rate of LNM", fontsize=12)
        ax[1, 0].legend(loc='lower right', fontsize=9)
        ax[1, 0].set_title('C', fontdict= {'family':'Arial','weight': 'semibold', 'fontsize': 12}, x=-0.03,y=1.03)
        index = index + 1
    # plt.show()



    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = dc.calculate_net_benefit_model(thresh_group, yProbas[0], yReals[0])
    net_benefit_all = dc.calculate_net_benefit_all(thresh_group, y_test)
    net_benefit_model1 = dc.calculate_net_benefit_model(thresh_group, yProbas[2], yReals[2])
    net_benefit_all1 = dc.calculate_net_benefit_all(thresh_group, yReals[2])
    net_models = [net_benefit_model, net_benefit_model1]
    net_alls = [net_benefit_all, net_benefit_all1]
    nases = ['ResNet', 'Radiomics']

    # fig, ax = plt.subplots()
    ax[1, 1] = dc.plot_DCAs(ax[1, 1], thresh_group, net_models, net_alls, colors, nases)
    ax[1, 1].set_title('D', fontdict= {'family':'Arial','weight': 'semibold', 'fontsize': 12}, x=-0.03, y=1.03)


    plt.show()



def GetMetrics(true_lab, prediction, thres=None):
    auc = roc_auc_score(y_true=true_lab, y_score=prediction)

    if thres:
        hlab = np.zeros(len(prediction))
        hlab[np.array(prediction) > thres] = 1
    else:
        hlab = np.round(prediction)

    cm = confusion_matrix(y_true=true_lab, y_pred=hlab)
    tn = cm[0, 0]
    fn = cm[1, 0]
    tp = cm[1, 1]
    fp = cm[0, 1]
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tn + fn + tp + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)


    return auc, acc, sen, spec, ppv, npv, cm


def featureSelect(featuresMetods, X_, y, file_outpath):
    strTemp = ''
    for key, value in featuresMetods.items():
        if value:
            if strTemp == '':
                strTemp = key
            else:
                strTemp = strTemp + " + " + key
    print('feature_selection methods:' + strTemp)
    X_result = None

    # Univariate feature selection by F-score 
    if featuresMetods['Univariate']:
        print('Univariate features selected-----')
        from sklearn.feature_selection import SelectPercentile, f_classif

        sel_ = SelectPercentile(f_classif, percentile=20).fit(X_, y)
        features_to_keep = X_.columns[sel_.get_support()]
        X_univ = sel_.transform(X_.fillna(0))
        X_univ = pd.DataFrame(X_univ)
        X_univ.columns = features_to_keep
        X_univ.index = X_.index
        X_ = X_univ
        X_result = X_univ

        print('univariate analysis: ', X_univ.shape)

    if featuresMetods['Lasso']:
        print('Lasso features selected-----')

        m_log_alphas, model, mp_coef, X_lasso = lassoFeatures(X_, y, file_outpath)
        if X_lasso is None:
            exit()

        X_ = X_lasso.iloc[:, 1:-1]
        print('lasso:', X_.shape)
        X_result = X_


    if featuresMetods['ElasticNet']:
        print('ElasticNet features selected-----')

        m_log_alphas, model, mp_coef, X_elasticNet = elasticNetFeatures(X_, y, file_outpath)
        if X_elasticNet is None:
            exit()
        X_ = X_elasticNet.iloc[:, 1:-1]
        print('ElasticNet:', X_.shape)
        X_result = X_


    if featuresMetods['RFE']:
        print('RFE features selected-----')

        model = LogisticRegression()
        sel_ = RFE(model, 15)
        X_rfe = sel_.fit_transform(X_, y)
        features_to_keep = X_.columns[sel_.get_support()]
        X_rfe = pd.DataFrame(X_rfe)
        X_rfe.columns = features_to_keep

        X_ = X_rfe
        print('RFE:', X_.shape)
        X_result = X_

    # Recursive feature addition
    if featuresMetods['RFA']:
        print('RFA features selected-----')
        model_all_features = DecisionTreeClassifier(random_state=rd)
        model_all_features.fit(X_, y)
        y_pred_ = model_all_features.predict_proba(X_)[:, 1]

        features = pd.Series(model_all_features.feature_importances_)
        features.index = X_.columns
        features.sort_values(ascending=False, inplace=True)
        features = list(features.index)

        model_one_feature = SVC(kernel='rbf', random_state=0, probability=True)  # ori
        model_one_feature.fit(X_[features[0]].to_frame(), y)
        y_pred_ = model_one_feature.predict_proba(X_[features[0]].to_frame())[:, 1]
        auc_score_first = roc_auc_score(y, y_pred_)

        print('doing recursive feature addition')
        features_to_keep = [features[0]]
        count = 1
        for feature in features[1:]:
            print()
            print('testing feature: ', feature, ' which is feature ', count,
                  ' out of ', len(features))
            count = count + 1
            model_int = SVC(kernel='rbf', random_state=rd, probability=True)

            model_int.fit(
                X_[features_to_keep + [feature]], y)

            y_pred_ = model_int.predict_proba(
                X_[features_to_keep + [feature]])[:, 1]

            auc_score_int = roc_auc_score(y, y_pred_)
            print('New Test ROC AUC={}'.format((auc_score_int)))
            print('All features Test ROC AUC={}'.format((auc_score_first)))

            diff_auc = auc_score_int - auc_score_first

            if diff_auc >= 0.02:
                print('Increase in ROC AUC={}'.format(diff_auc))
                print('keep: ', feature)
                print

                auc_score_first = auc_score_int
                features_to_keep.append(feature)
            else:
                print('Increase in ROC AUC={}'.format(diff_auc))
                print('remove: ', feature)
                print
        print('DONE!!')
        print('total features to keep: ', len(features_to_keep))
        print(features_to_keep)
        X_selectFeature = X_[features_to_keep]
        X_result = X_selectFeature
        return X_selectFeature

    return X_result


def ROCC(file1,file2):
    res = pd.read_csv(file1)
    rad = pd.read_csv(file2)
    y_pred_test = res['y_pred_test']
    y_pred_train = res['y_pred_train']
    y_test = res['y_test']
    y_train = res['y_train']

    y_pred_test1 = rad['y_pred_test']
    y_pred_train1 = rad['y_pred_train']
    y_test1 = rad['y_test']
    y_train1 = rad['y_train']


    yProbas = [y_pred_test,
               y_pred_train,
               y_pred_test1,
               y_pred_train1]
    yReals = [y_test, y_train, y_test1, y_train1]

    names = ['Testing cohort: ResNet',
             'Training cohort: ResNet',
             'Testing cohort: Radiomics',
             'Training cohort: Radiomics'
             ]

    colors = ['crimson',
              'steelblue',
              'steelblue',
              'steelblue',
              'mediumseagreen',
              'mediumpurple'
              ]
    linestyles = ['--', '-', '--', '-']


    ROC_CaCurv_(names, yProbas, colors, linestyles, yReals)





if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    rd = 119
    file_inpath = '../data/'
    file_outpath = '../data/selectedFeatures/'


    filelists = os.listdir(file_inpath)
    filelists = ['clin_a.csv',
                 'clin_.csv',
                 'harmonized_Radiomics.csv',
                 'Features_Resnet.csv',
                 'Features_InceptionResNetV2.csv',
                 'Features_Resnet_FC.csv',  # 5
                 'Features_Resnet_res2c.csv',
                 'Features_Resnet_res3d.csv',
                 'Features_Resnet_resf4.csv',
                 'Features_VGG16.csv', # 9
                 'Features_VGG19.csv',
                 'Features_Xception.csv',
                 'Features_Radiomics_Res.csv'
                 ]

    featuresMetods = {'Univariate':  True,
                      'Lasso': False,
                      'ElasticNet': False,
                      'RFE': False,
                      'RFA': True}

    data_c_all = pd.read_csv(os.path.join(file_inpath, filelists[1]))
    data_c = data_c_all.loc[:, ['ID', 'P_N']]

    data_r = pd.read_csv(os.path.join(file_inpath, filelists[11]))
    print('read_data：'+ filelists[3])

    data_c_r = pd.merge(data_r, data_c, on='ID')
    data_c_r.fillna(0)
    print('input_data：', data_c_r.shape)
    data_c_r.to_csv(os.path.join(file_outpath, 'data_R_clin.csv'))
    X_original = data_c_r.iloc[:,1:-1]

    colNames = data_r.columns[1:]
    X_original = X_original.fillna(0)
    X_original = X_original.astype(np.float64)
    X_std = StandardScaler().fit_transform(X_original)  
    X_std = pd.DataFrame(X_std)
    X_std.columns = colNames
    y = data_c['P_N']
    y_orignal = y
    X_ = X_std
    print('all: ' + str(len(X_)))
    print(y.value_counts())


    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.3, random_state=24)#1
    X_all = X_
    y_all = y


    X_all_ = featureSelect(featuresMetods, X_all, y_all, file_outpath)
    X_train = X_train[X_all_.columns]
    X_test = X_test[X_all_.columns]


    #MLSVM(X_train, y_train, X_test, y_test,1)
    machineLearning(X_train, y_train, X_test, y_test)
   








