import pandas as pd
from catboost import cv, Pool, CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from optuna.samplers import RandomSampler, TPESampler
from catboost.utils import eval_metric

import matplotlib.pyplot as pyplot
import optuna
from IPython.display import display
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier


class Catboost_classificator:

    @staticmethod
    def catboost_base_model_func(X_train,
                         y_train,
                         X_val,
                         y_val,
                         cat_feature,
                         params={}):
        # создаем объект класса катбуст с дефолтными параметрами
        base_model = CatBoostClassifier(
                                        random_state=0,
                                        use_best_model = True,
                                        cat_features=cat_feature,
                                        verbose = 50,
                                        **params
                                       )

        # обучаем катбуст
        base_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        return base_model

    @staticmethod
    def objective(trial, train_pool, val_pool, exist_params=None, param=None):
        '''param - текущий оптимизируеый параметр'''
        # объявляем все вариантиы параметров
        params = {
            'grow_policy': {'type': str, 'cats': ['SymmetricTree', 'Lossguide', 'Depthwise']},
            'max_depth': {'type': int, 'start': 2, 'end': 8, 'step': 1},
            'l2_leaf_reg': {'type': int, 'start': 1, 'end': 10, 'step': 1},
            'colsample_bylevel': {'type': float, 'start': 0.1, 'end': 1, 'step': 0.1},  # float?
            'min_data_in_leaf': {'type': int, 'start': 2, 'end': 20, 'step': 1},
            'one_hot_max_size': {'type': int, 'start': 2, 'end': 20, 'step': 1},
        }
        # выбираем текущий параметр и его значения для оптимизации
        display(param)
        # exist_params - уже определенные параметры, с которыми будет работать модель
        if not exist_params:
            exist_params = {}
        # в зависимости от типа параметра используем нужную функцию optuna
        if params[param]['type'] == str:
            param_optuna = {param: trial.suggest_categorical(param, params[param]['cats'])}
        elif params[param]['type'] == int:
            param_optuna = {param: trial.suggest_int(param, params[param]['start'], params[param]['end'],
                                                     step=params[param]['step'])}
        else:
            param_optuna = {param: trial.suggest_float(param, params[param]['start'], params[param]['end'],
                                                       step=params[param]['step'])}

        # создаем модель по выбранным ранее параметрам + ОДНОМУ, тестирующимуся в данный момент
        model = CatBoostClassifier(eval_metric='AUC', **param_optuna, **exist_params, random_seed=0,
                                   auto_class_weights='Balanced')
        model.fit(train_pool, verbose=0, eval_set=val_pool, use_best_model=False)
        y_pred = model.predict_proba(val_pool)[:, 1]
        # возвращаем AUC на valid множестве по данной модели
        #     y_pred = model.predict(val_pool)
        return eval_metric(val_pool.get_label(), y_pred, 'AUC')

    @staticmethod
    def catboost_optuna_model(
            X_train,
            y_train,
            X_val,
            y_val,
            cat_feature,
            optuna_n_trials=300,
            build_early_stopping_rounds=30,
            plot_calibration=True,
            calibrate=True,
            calibrate_way=1
    ):
        """
        Построение модели
        cat_feature - набор категориальных фичей

        """

        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feature, feature_names=list(X_train.columns))
        val_pool = Pool(data=X_val, label=y_val, cat_features=cat_feature, feature_names=list(X_val.columns))
        build_pool = Pool(data=pd.concat([X_train, X_val]), label=pd.concat([y_train, y_val]), cat_features=cat_feature,
                          feature_names=list(X_train.columns))

        base_params = {}
        # лучший auc на valid множестве
        best_auc = 0

        print('ПОДБОР БАЗОВЫХ ПАРАМЕТРОВ')
        # набор тестируемых значений learning rate
        for learning_rate in [0.2, 0.05, 0.02, 0.04, 0.1, 0.01, 0.001, 0.0001]:  # [0.05]:  #0.001, 0.01,  0.06, 0.0001
            change = False
            cross_validation = cv(pool=build_pool,
                                  params={'learning_rate': learning_rate, 'loss_function': 'Logloss',
                                          'eval_metric': 'AUC'},
                                  iterations=1000,
                                  fold_count=3,  # The number of folds to split the dataset into.
                                  partition_random_seed=0,
                                  logging_level='Silent',
                                  stratified=True,
                                  plot=True,
                                  early_stopping_rounds=build_early_stopping_rounds,
                                  return_models=True
                                  )  # построит fold_count моделей,
            '''
            например на 1-ой модели будет 145 деревье, на второй 135, на 3-ий модели 150 моделей, на 10-ой модели 150 деревь

            '''

            # усредненные данные по выборкам (1 на модель на каждый fold) в.т.ч - средний AUC на каждой итерации
            df_cv = cross_validation[0]
            print('результат-датафрейм cv')
            display(df_cv, '-' * 80)
            # итерация(первая, если AUC одинаков на нескольких до остановки) с лучшим AUC
            best_current_iterations = df_cv[df_cv['test-AUC-mean'] == df_cv['test-AUC-mean'].max()]['iterations'].iloc[
                                          0] + 1  # ( Добавляем 1 так как в catbosst количество деревьев считается с 0, где 0 это 1 дерево, iclod идет по индексам)
            print('текущее лучшее iterations с learning_rate:', learning_rate, ' - iterations:',
                  best_current_iterations)
            print([m._tree_count for m in cross_validation[1]])
            # по полученным параметрам кросс-валидации строим модель и проверяем метрику на valid-выборке
            model = CatBoostClassifier(eval_metric='AUC', auto_class_weights='Balanced',
                                       iterations=best_current_iterations, learning_rate=learning_rate, random_state=0,
                                       use_best_model=False)
                
            print('лог обучения модели на параметрах с cv')
            model.fit(train_pool, eval_set=val_pool)
            # AUC-predict_proba
            y_pred = model.predict_proba(val_pool)[:, 1]
            auc = eval_metric(val_pool.get_label(), y_pred, 'AUC')[0]
            print('tree_count_ на итерации', model.tree_count_)
            print('iterations на итерации', best_current_iterations)
            print('auc на итерации:', auc)

            if auc > best_auc:
                change = True
                best_auc = auc
                base_params['learning_rate'] = learning_rate
                base_params['iterations'] = model.tree_count_
            print('base_params на конец итерации:', base_params)

            if change:
                print('число итераций и learning_rate были переопределены на текущей итерации')
        # вывести конфигурацию catboost по деревьям и learning_rate
        print('ИТОГ ПОДБОРА БАЗОВЫХ ПАРАМЕТРОВ')
        print('best_auc', best_auc)
        print(base_params)
        print('ПОДБОР OPTUNA ПАРАМЕТРОВ')
        # оптимизировать количетсво итераций optuna
        optuna_n_trials = 50  # на каждый

        params = ['grow_policy', 'max_depth', 'l2_leaf_reg']
        
        for param in params:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=0))
            study.optimize(lambda trial: Catboost_classificator.objective(trial,
                                                                 train_pool,
                                                                 val_pool,
                                                                 exist_params=base_params,
                                                                 param=param
                                                                 ), n_trials=optuna_n_trials
                           )
            
            model = CatBoostClassifier(eval_metric='AUC', auto_class_weights='Balanced', random_state=0, **base_params,
                                       **{param: study.best_params[param]}, use_best_model=False)
           
            print('лог обучения модели на параметрах optuna')
            model.fit(train_pool, eval_set=val_pool)
            y_pred = model.predict_proba(val_pool)[:, 1]
            eval_auc = eval_metric(val_pool.get_label(), y_pred, 'AUC')[0]
            print('_______')
            display(val_pool.get_label())
            display(y_pred)
            auc = roc_auc_score(val_pool.get_label(), y_pred)
            display(auc)
            display(eval_auc)

            # сохраняем параметр только при улучшении метрики AUC
            if auc > best_auc:
                print('best_auc -', best_auc, 'auc с новым параметром', param, ' -', auc,
                      '=> добавляем в список параметров')

                best_auc = auc
                base_params[param] = study.best_params[param]
                
        print('ИТОГ ПОДБОРА OPTUNA ПАРАМЕТРОВ')
        display(base_params)
        model = CatBoostClassifier(
                eval_metric='AUC',
                **base_params,
                use_best_model=False,
                auto_class_weights='Balanced',
                random_seed=0
            )
        
        print('best_auc', best_auc)
        print('лог обучения финальной модели')
        model.fit(
            train_pool,
            eval_set=val_pool,
            plot=True
        )
        print('tree_count_ финальной модели', model.tree_count_)

        # калибровка
        '''
        способо 1 это строим на train_pool и показываем на val_pool на статусах
        способо 2 на  val_pool делим на 2 выборки 80/20 строим модель и показываем

        определить модель
        кажется логит регрессия
        если задам параметро PlotlyColubration = True то выводит график curve до калибровки и после
        '''

        if calibrate:
            # если calibrate_way == 1 - то калибруем по train и проверяем на valid
            # если calibrate_way == 2 - то калибруем по valid(80%) и проверяем на valid(20%)
            if calibrate_way in [2, 4]:
                ValSample = X_val.copy()
                ValSample['y'] = y_val

                calibrate, valid = train_test_split(ValSample, train_size=0.8, stratify=ValSample['y'], random_state=22)
                X_val = valid.drop(columns='y')
                y_val = valid['y']
                X_calibrate = calibrate.drop(columns='y')
                y_calibrate = calibrate['y']

            if plot_calibration:
                print('CALIBRATION DISPLAY')
                # predict probabilities
                probs = model.predict_proba(X_val)[:, 1]
                # reliability diagram
                fop, mpv = calibration_curve(y_val, probs, n_bins=10)
                # plot perfectly calibrated
                pyplot.plot([0, 1], [0, 1], linestyle='--')
                # plot model reliability
                pyplot.plot(mpv, fop, marker='.')
                pyplot.show()

            if calibrate_way in [1, 2]:
                calibrated_clf = CalibratedClassifierCV(model, cv="prefit", method = 'isotonic')
                if calibrate_way == 1:
                    calibrated_clf.fit(X_train, y_train)
                elif calibrate_way == 2:
                    calibrated_clf.fit(X_calibrate, y_calibrate)
            elif calibrate_way in [3, 4]:
                grid = {
                    "C": [100, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
                    "penalty": ["l2", 'elasticnet'],
                    'class_weight': ['balanced'],  # Использовать только balanced, иначе ставятся только 0
                }
                logreg_cv = LogisticRegression()
                lr_model = GridSearchCV(logreg_cv, grid, cv=3, verbose=1)
                if calibrate_way == 3:
                    lr_model.fit(model.predict_proba(X_train)[:, [1]], y_train)

                elif calibrate_way == 4:
                    lr_model.fit(model.predict_proba(X_calibrate)[:, [1]], y_calibrate)

            if plot_calibration:
                print('CALIBRATION DISPLAY _AFTER CALIBRATION')
                # predict probabilities
                if calibrate_way in [1, 2]:
                    probs = calibrated_clf.predict_proba(X_val)[:, 1]
                elif calibrate_way in [3, 4]:
                    probs = lr_model.predict_proba(model.predict_proba(X_val)[:, [1]])[:, 1]
                # reliability diagram
                fop, mpv = calibration_curve(y_val, probs, n_bins=10)
                # plot perfectly calibrated
                pyplot.plot([0, 1], [0, 1], linestyle='--')
                # plot model reliability
                pyplot.plot(mpv, fop, marker='.')
                pyplot.show()
            if calibrate_way in [1, 2]:
                return calibrated_clf
            return lr_model, model
        return model
    
