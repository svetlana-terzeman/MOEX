import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from catboost import cv, Pool
from scipy import stats
from scipy.stats import ks_2samp
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    auc
)
import os
import shutil
from catboost import cv, Pool, CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split

class BinaryClassificationEvaluator:
    def __init__(self, y_true, y_pred, y_pred_prob):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob

    def plot_roc_curve(self, path, verbose = True):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc_score(self.y_true, self.y_pred_prob):.4f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if verbose:
            plt.savefig(f'{path}/plot_roc_curve.png')
        else:
            plt.show()
        plt.close()

    def plot_precision_recall_curve(self, path, verbose = True):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        if verbose:
            plt.savefig(f'{path}/plot_precision_recall_curve.png')
        else:
            plt.show()
        plt.close()

#     def plot_confusion_matrix(self, path):
        
       
#         cm = confusion_matrix(self.y_true, self.y_pred)
#         # Нормализация матрицы ошибок
#         cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         plt.figure(figsize=(10, 6))
#         sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.title('Confusion Matrix')
        
#         plt.savefig(f'{path}/plot_confusion_matrix.png')
#         plt.close()
        
# #         plt.show()

    def plot_confusion_matrix(self, path):
    
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 5))

        # Первая тепловая карта: абсолютные значения
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Absolute)')

        # Вторая тепловая карта: нормализованные значения
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Normalized)')

        plt.tight_layout()
        plt.savefig(f'{path}/plot_confusion_matrix.png')
        plt.close()
    
            
        
    def plot_calibration(self, path, verbose = True):
        
        tiles = np.percentile(self.y_pred_prob, np.linspace(0, 100, 10))
        buckets = pd.cut(self.y_pred_prob, tiles, duplicates='drop')
        pred_df = pd.DataFrame({'buckets':buckets, 'y_pred':self.y_pred_prob, 'y_test':self.y_true})
        for_test = pred_df.groupby('buckets', as_index=False).apply(lambda x: pd.Series({'pred_mean' : x['y_pred'].mean(),
                                                                              'fact_mean' : x['y_test'].mean(),
                                                                              'pred_std_0': proportion_confint(
                                                                                                count=x['y_test'].sum(),
                                                                                                nobs=x.shape[0],
                                                                                                alpha=0.05,
                                                                                                method='wilson'
                                                                                                )[0],
                                                                              'pred_std_1': proportion_confint(
                                                                                                count=x['y_test'].sum(),
                                                                                                nobs=x.shape[0],
                                                                                                alpha=0.05,
                                                                                                method='wilson'
                                                                                            )[1],
                                                                              'cnt' : x.shape[0]}))
           
        fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
        ax.tick_params(labelrotation=45)
        # use of a float for the position:
        ax2 = ax.twinx()
        ax2.bar(for_test['buckets'].astype(str), for_test['cnt'], alpha=0.3, color='green')
        ax.fill_between(for_test['buckets'].astype(str), y1=for_test['pred_std_0'], y2=for_test['pred_std_1'], alpha=0.5)
        #plt.scatter(for_test['buckets'].astype(str), y=for_test['pred_mean'])
        ax.scatter(for_test['buckets'].astype(str), y=for_test['fact_mean'])
        fig.suptitle('Калибровочная кривая с дов. интервалом')

        if verbose:
            plt.savefig(f'{path}/plot_calibration.png')
        else:
            plt.show()
        plt.close()
       

        
def matrix(X_test, y_test, clf, path, class_percent=0.6, n_steps = 0.05):
    """
        Функция построения матрицы ошибок
        Параметры:
            X_test: Тестовая выборка с фичамм
            y_test: Таргет
            clf: Модель классификатора
            class_percent: threshold для вероятностей
    """
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_binary = [1 if k >= class_percent else 0 for k in y_pred_proba]
    
    evaluator = BinaryClassificationEvaluator(np.array(y_test), np.array(y_pred_binary), np.array(y_pred_proba))
#     evaluator.evaluate()
    
    evaluator.plot_roc_curve(path)
    evaluator.plot_precision_recall_curve(path)
#     evaluator.plot_confusion_matrix(path)
    # evaluator.plot_optimal_proba(path)
    evaluator.plot_calibration(path)
    evaluator.plot_total_stata(path, N = n_steps)
    #return evaluator.plot_total_stata(N = n_steps)

    #print(f'ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}')

    # смотрим матрицу ошибок
#     conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_binary)
#     vis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
#     vis.plot()
#     plt.grid(False)
#     plt.title('Матрица ошибок')
#     plt.show()

#     cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

#     # Plot normalized confusion matrix
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[0, 1])
#     disp.plot(cmap=plt.cm.Blues)  # You can specify your preferred colormap
#     plt.title('Normalized Confusion Matrix')
#     plt.show()

def best_threshold(clf, X_test, y_test_binary):
    """
        Функция получения наилучшего threshold для конкретной модели
        Параметры:
            clf: Модель классификатора
            X_test: Тестовая выборка с фичамм
            y_test_binary: Таргет
    """
    
    # Получаем вероятности предсказаний
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Вычисляем значения FPR и TPR для разных порогов
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)

    # Находим индекс порога, который дает максимальное значение TPR и минимальное значение FPR
    best_threshold_index = np.argmax(tpr - fpr)

    # Получаем порог
    best_threshold = thresholds[best_threshold_index]
    print("Лучший порог:", best_threshold)

    # Оцениваем модель с использованием найденного порога
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    # Вычисляем ROC AUC для оценки качества модели с лучшим порогом
    roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
    print("ROC AUC:", roc_auc)
    
def total_stata(y_test, y_test_binary, y_pred_proba, N=0.05, cost_per_user=180, treshold = 0.5):
    
    """
        Функция получения общей статистики по классификатору
        Параметры:
            clf: Модель классификатора
            y_test: вещественные значения таргета
            y_test_binary: Таргет(бинарный)
            y_pred_proba: вероятности таргетов
            N: шаг, для деления выборки на интервалы
    """
    
    df_r = pd.DataFrame({
        'y_test': y_test,
        'y_test_binary': y_test_binary,
        'y_pred_proba': y_pred_proba

    })
    all_loans = df_r.shape[0]
    df_r['interval'] = pd.cut(df_r['y_pred_proba'], bins=np.arange(0, 1 + N, N)) # ,  include_lowest = True
    
    
    
    df_r['interval'] = df_r['interval'].astype(str)
    df_r['approved'] = df_r.apply(lambda x: 1 if x['y_pred_proba'] >= treshold  else 0, axis=1)
    #display(df_r)
    df_r = df_r.drop(columns=['y_pred_proba'])
    
    df_r = df_r.groupby(['interval']).agg({'y_test': ['mean', 'count','sum'], 'y_test_binary': 'mean', 'approved':'min'}).reset_index()
    
    df_r.columns                        = ['_'.join(k) for k in df_r.columns.ravel()]
    df_r['y_test_costs']                = df_r['y_test_count'] * cost_per_user
    df_r['y_test_costs_reverse_cumsum'] = df_r['y_test_costs'][::-1].cumsum()[::-1]
    df_r['y_test_sum_reverse_cumsum']   = df_r['y_test_sum'][::-1].cumsum()[::-1]
    df_r['profit'] = df_r['y_test_sum_reverse_cumsum'] - df_r['y_test_costs_reverse_cumsum']
    #display(df_r)
    
    
    #df_r = df_r.drop(columns=['y_test_binary_mean'])
    #df_r.columns = ['Скор NGR', 'Средний % NGR', 'Кол-во клиентов']
    df_r = df_r.rename(columns={
        'interval_':'Скор NGR',
        'y_test_mean':'Средний NGR',
        'y_test_count':'Кол-во клиентов',
        'y_test_sum':'Сумма поступлений',
        'y_test_binary_mean':'Доля юзеров с профитом',
        'y_test_costs':'Сумма издержек',
        'y_test_costs_reverse_cumsum':'Накопительная сумма издержек',
        'y_test_sum_reverse_cumsum':'Накопительная сумма поступлений',
        'profit':'Профит',
    })
    df_r['Доля'] = df_r['Кол-во клиентов'] / all_loans
    
    df_r['Накопительная доля'] = df_r['Доля'][::-1].cumsum()[::-1]
    df_r['Накопительное кол-во клиентов'] = df_r['Кол-во клиентов'][::-1].cumsum()[::-1]
    
    
    return df_r

def find_feature_importance(data: pd.DataFrame, model, path):
    '''
        Функция поиска features importances
        data: pd.DataFrame
            Тренировочная выборка данных
        model: CatBoost
            Модель классификатора
    
    '''
    
    importance = pd.DataFrame({
        'features': data.columns,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values(by='importance')

#     importance.sort_values(by='features', inplace=True)
    
    plt.figure(figsize=(20, 8))
    plt.barh(importance['features'], importance['importance'])
    plt.title(f'Feature Importance')
    plt.tight_layout()
#     plt.show()
    plt.savefig(f'{path}/feature_importance.png')
    plt.close()
    
    return importance


    
# def find_feature_importance_catboost(X_train, y_train, cat_features, model, type_):
#     # Создание пула данных для CatBoost
#     pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    
#     # Получение важности признаков
#     importance = pd.DataFrame({
#         'features': X_train.columns,
#         'importance': model.get_feature_importance(pool, type=type_)
#     }).sort_values(by='importance', ascending=False)
    
#     # Определение цветовой палитры с уникальным цветом для каждого признака
#     palette = sns.color_palette("husl", len(importance))  # Можно заменить на "tab20", "Set3" и др.
    
#     # Построение графика
#     plt.figure(figsize=(20, 8))
#     sns.barplot(
#         y=importance['features'],
#         x=importance['importance'],
#         palette=palette
#     )
#     plt.title(type_, fontsize=16)
#     plt.xlabel('Importance', fontsize=14)
#     plt.ylabel('Features', fontsize=14)
#     plt.grid(axis='x', linestyle='--', alpha=0.7)
#     plt.show()
#     plt.close()

#     return importance


def sampling_test(df, clf, fraction, N, categorical_features, stratify, class_percent, path, verbose = False):
    
    """
    Возвращает датафрейм с N тестами.

    :param df: Исходный DataFrame
    :param df: Модель классификатора
    :param fraction: Доля строк для выборки (от 0 до 1)
    :param N: кол-во тестов(итераций)
    :param stratify: применять стратицикацию к выборки или нет
    :return: DataFrame с случайной выборкой строк
    """
    fraction = 1-fraction # инверсия для формирования тестовой выборки
    all_result = pd.DataFrame()
    for iteration in range(1, N+1):
        
        if stratify:
            random_test, x_ = train_test_split(df, stratify=df['bin_type'], test_size=fraction, shuffle=True)
        else:
            random_test, x_ = train_test_split(df, test_size=fraction, shuffle=True)
#         if len(categorical_features) > 0:
#             random_test[categorical_features] = random_test[categorical_features].astype(str)
        
    
        X_test = random_test.drop('bin_type', axis = 1)
        y_test = random_test['bin_type']
        
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_binary = [1 if k >= class_percent else 0 for k in y_pred_proba]
        
        
        
        result = {
            f'Итерация': iteration,
            'Dataset_size':X_test.shape[0],
            'Target mean':y_test.mean(),
            'Accuracy': accuracy_score(y_test, y_pred_binary),
            'Precision': precision_score(y_test, y_pred_binary),
            'Recall': recall_score(y_test, y_pred_binary),
            'F1 Score': f1_score(y_test, y_pred_binary),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            
        }
        
        result = pd.DataFrame(result, index=[0])
        all_result = pd.concat((all_result, result), axis = 0)
    if 'interval' in path:
        all_result.describe().round(3).to_csv(f"{path}")
    if verbose:
        display(all_result.describe().round(3))
    return all_result


# def find_best_threshold(X_test, y_test, model):
    
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
#     precisions, recalls, thresholds = precision_recall_curve(y_test,  y_pred_proba)  
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
#     f1_scores = np.nan_to_num(f1_scores, nan=0.0) # если f1 получила nan Значение
#     optimal_threshold = round(thresholds[np.argmax(f1_scores)], 2)
    
#     return optimal_threshold
    
def get_metrics(result_data, info_data, X_test, y_test, model, dict_sampling_test, path, class_percent=0.6, n_steps = 0.05, verbose_metrics = False):
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_binary = [1 if k >= class_percent else 0 for k in y_pred_proba]
    y_pred = model.predict(X_test)
    
    # Поиск оптимального порога по F1-score
    precisions, recalls, thresholds = precision_recall_curve(y_test,  y_pred_proba)  
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0) # если f1 получила nan Значение
    optimal_f1_score = np.max(f1_scores)

    a = BinaryClassificationEvaluator(np.array(y_test), np.array(y_test_binary), np.array(y_pred_proba))
    table_stats = a.plot_total_stata(path = path, verbose = verbose_metrics)
    table_stats['tag'] = table_stats['Скор'].apply(lambda x: 1 if class_percent in x else 0)
    table_stats = table_stats[table_stats['tag'] == 1]
    
    dict_sampling_test['path'] = path
    samples_test = sampling_test(**dict_sampling_test)
    weighted_f1 = samples_test = samples_test.describe()['F1 Score']['mean']
    
    metrics = {
            'Examples':X_test.shape[0],
            'Accuracy': accuracy_score(y_test, y_test_binary),
            'Precision': precision_score(y_test, y_test_binary),
            'Recall': recall_score(y_test, y_test_binary),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'F1 Score': f1_score(y_test, y_test_binary),
            'Optimal F1-Score': optimal_f1_score,
            'Weighted F1 Score': weighted_f1,
            'Optimal Threshold': class_percent,
            'Сохранение потока': round(table_stats['Накопительная доля от общего кол-ва'].iloc[0], 3),
            'Фрод': round(1 - table_stats['Накопительная доля хороших'].iloc[0], 3)
        }
    
    df_ = pd.DataFrame({**info_data, **metrics, **{'Дата формирования отчета': pd.Timestamp.now().date()}}, index=[0])
    result_data = pd.concat((result_data, df_), axis = 0).reset_index(drop=True).round(2)
                        
    return result_data


def perform_rfecv(estimator, X_train, y_train, important_features, categorical_features, cv_splits=2, step=1, scoring='roc_auc'):
    
    X_train = X_train.drop(categorical_features, axis = 1)
    if estimator == 'catboost' or estimator == 'catboostoptuna':
        model = CatBoostClassifier(eval_metric='AUC',
                                    auto_class_weights='Balanced',
                                    random_state=0,
                                    loss_function='Logloss'
                                    
                                       )
    if estimator == 'xgboost' or estimator == 'xgboostoptuna':
        model = XGBClassifier(eval_metric='auc',
                                     random_state=0,
                                     loss_function='Logloss',
                                    )
    # Настройка RFECV
    rfecv = RFECV(estimator=model, step=step, cv=StratifiedKFold(cv_splits), scoring=scoring)
    # Выполнение отбора признаков
    rfecv.fit(X_train, y_train)
    
    #Отбор новых признаков
    selected_features = rfecv.support_
    selected_features = list(X_train.columns[selected_features])
    # учитываем отобранные фичи и фичи которые не могут быть удалены
    selected_features = list(set(selected_features + important_features + categorical_features))
    return selected_features


def intervals(test, bins_method, bins_number, bets_feature, model_data, result_metrics, model, threshold, sampling_params, path, verbose_metrics = False):
    
    test_sample = test.copy()
    test_sample = pd.concat((test_sample, bets_feature), axis = 1)
    # разбиваем выборку на интервалы
    if bins_method == 'quantile':
        test_sample['group_tag'] = pd.qcut(test_sample[bets_feature.name], q=bins_number, duplicates='drop')
    if bins_method == 'interval':
        test_sample['group_tag'] = pd.cut(test_sample[bets_feature.name], bins=bins_number, duplicates='drop')
    
    print(f'{bets_feature.name} разбиение по методу {bins_method}') 
    # проходимся по каждому интервалу
    for tag in test_sample['group_tag'].sort_values().unique():
        print('интервал: ',  tag)
        # создаем копию словаря с инфой о модели
        inter_model_data = model_data.copy()
        # создаем копию словаря с инфой для sampling_test
        inter_sampling_params = sampling_params.copy()
        # меняем название выборки
        inter_model_data['Выборка'] = inter_model_data['Выборка'] + f'_{bins_method}_{tag}'
        # отбираем данные за конкретный интервал
        interval_sample = test_sample[test_sample['group_tag'] == tag].drop(['group_tag', bets_feature.name], axis = 1)
        # разбиваем выборку на признаки/таргет
        x_test_, y_test_ = interval_sample.drop(['bin_type'], axis = 1), interval_sample['bin_type']
        
        # заменяем тестовый df на df текущего интервала 
        inter_sampling_params['df'] = interval_sample
        # ставим stratify True, так как данных меньше в выборке и при делениии в sampling_test
        # могут быть случаи когда будет только один класс - ОШИБКА при подсчете метрик
        inter_sampling_params['stratify'] = True
        
        path_ = f'{path}/intervals/interval_{tag}_stata.csv'
        # вызываем функцию расчета метрик для конкретного интервала
        result_metrics = get_metrics(
            result_data = result_metrics, 
            info_data = inter_model_data, 
            X_test = x_test_, 
            y_test = y_test_, 
            model = model,  
            dict_sampling_test = inter_sampling_params,
            class_percent = threshold,
            verbose_metrics = verbose_metrics,
            path = path_
        )
        
    return result_metrics

def clear_only_files(folder_path):
    # Проверяем, существует ли папка
    if os.path.exists(folder_path):
        # Проходим по всем файлам и папкам внутри указанной папки
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Удаляем файл или ссылку
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Не удалось удалить {file_path}. Причина: {e}')
    else:
        print(f'Папка {folder_path} не существует')
        

def error_matrix(X_test, y_test, clf, class_percent=0.6, path=None):
    """
    Функция построения матрицы ошибок
    Параметры:
        X_test: Тестовая выборка с фичами
        y_test: Таргет
        clf: Модель классификатора
        class_percent: threshold для вероятностей
        type_: если None -> вывод на экран, иначе сохранение в PNG с префиксом
    """
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_binary = [1 if k >= class_percent else 0 for k in y_pred_proba]

    print(f'ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}')

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_binary)
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Обычная матрица
    disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
    disp1.plot(ax=axes[0], colorbar=False)
    axes[0].set_title('Матрица ошибок')
    axes[0].grid(False)

    # Нормализованная матрица
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=[0, 1])
    disp2.plot(ax=axes[1], cmap=plt.cm.Blues, colorbar=False)
    axes[1].set_title('Normalized Confusion Matrix')

    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()

    
def mean_absolute_percentage_error(y_true, y_pred):
    # MAPE-value
    # < 11% высокоточный прогноз
    # >= 11 & < 21 хороший прогноз
    # >= 21 & < 51 разумный (нормальный) прогноз
    # >= 51 неточный прогноз
    # исключаем из списка y_true значения с нулевыми показателем и индекс исключенного значени  исключаем из y_pred

    empty_idx = []  # индексы с пустыми значениями
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    empty_idx = np.nonzero(y_true == 0)[0]
    y_true = np.delete(y_true, empty_idx, axis=0)  # это numpy.ndarry
    y_pred = np.delete(y_pred, empty_idx, axis=0)  # это numpy.ndarry
    

    _coef = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    #print(_coef)

    return _coef

def calculate_columnwise_mape(df1, df2):
    # Инициализируем словарь для хранения MAPE по каждому столбцу
    mape_stats = {}
    common_columns = df1.columns.intersection(df2.columns)

    # Проходим по всем общим столбцам
    for col in common_columns:
        #print(col)
        # Получаем описательные статистики для каждого столбца
        if df1[col].dtype != 'object':
        
            desc1 = df1[col].describe(percentiles=[i for i in np.arange(0,1,0.05)]).iloc[6:-1]
            desc2 = df2[col].describe(percentiles=[i for i in np.arange(0,1,0.05)]).iloc[6:-1]
            common_indexes = desc1.index.intersection(desc2.index)
            #display(desc1)
            #display(desc2)
        else:
            desc1 = df1[col].value_counts(normalize=True)
            desc2 = df2[col].value_counts(normalize=True)
            common_indexes = desc1.index.intersection(desc2.index)
        
        # Рассчитываем MAPE для всех статистик
        mape = mean_absolute_percentage_error(desc1[common_indexes],desc2[common_indexes])
        #print(mape)
        

        # Сохраняем результат в словарь
        mape_stats[col] = mape
        
    
    # Преобразуем словарь в датафрейм
    mape_df = pd.DataFrame({'mape':mape_stats})

    return mape_df

def save_importances(df, path, file_name):
    df = df.sort_values(by='importance', ascending = False)
    plt.figure(figsize=(20, 8))
    ax = sns.barplot(y = df['features'], x = df['importance'])
    plt.title(f"{'FeatureImportance'} (in %)")
    plt.xlabel('Importance (%)')
    # Добавление значений на каждый столбик
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', label_type='edge')
    plt.savefig(f"{path}/{file_name}.png", bbox_inches='tight', dpi=150)
    
def find_best_threshold(y_true, y_pred_proba):
    
#     precisions, recalls, thresholds = precision_recall_curve(y_test,  y_pred_proba)  
#     f1_scores         = 2 * (precisions * recalls) / (precisions + recalls)
#     f1_scores         = np.nan_to_num(f1_scores, nan=0.0) # если f1 получила nan Значение
#     optimal_threshold = round(thresholds[np.argmax(f1_scores)], 2)

    # Вычисление TPR и FPR (ROC-кривая)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Вычисление True Negative Rate (TNR)
    tnr = 1 - fpr

    # Нахождение оптимального порога (где TPR и TNR пересекаются)
    intersection_idx = np.argmin(np.abs(tpr - tnr))
    optimal_threshold = thresholds[intersection_idx]
    optimal_tpr = tpr[intersection_idx]
    optimal_tnr = tnr[intersection_idx]
   
    return optimal_threshold, optimal_tpr, optimal_tnr, thresholds, fpr, tpr, tnr


def plot_roc_lift(y_true, y_pred_proba, path, verbose = True):
    # ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Lift-кривая
    precision, recall, thresholds_lift = precision_recall_curve(y_true, y_pred_proba)
    lift = precision / (sum(y_true) / len(y_true))  # Lift = Precision / Base Rate
    
    # --- График ROC-AUC ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    
    # --- График Lift-кривой ---
    plt.subplot(1, 2, 2)
    plt.plot(recall, lift, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Lift')
    plt.title('Lift Curve')
    plt.grid()

    plt.tight_layout()
    
    if verbose:
        plt.savefig(f'{path}/plot_roc_auc_lift_curves.png')
    
    else:
        plt.show()
    plt.close()

def plot_tpr_tnr_intersection(y_true, y_pred_proba, path, verbose = True):
    """
    Построение графика пересечения TPR и TNR и определение оптимального порога.
    
    Параметры:
        y_true (array-like): Истинные метки (0 или 1).
        y_pred_proba (array-like): Предсказанные вероятности.
    """

    
    optimal_threshold, optimal_tpr, optimal_tnr, thresholds, fpr, tpr, tnr = find_best_threshold(y_true, y_pred_proba)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr, label='True Positive Rate (TPR)', color='blue')
    plt.plot(thresholds, tnr, label='True Negative Rate (TNR)', color='green')
    plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.scatter(optimal_threshold, optimal_tpr, color='red', zorder=5)
    plt.scatter(optimal_threshold, optimal_tnr, color='red', zorder=5)
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Intersection of TPR and TNR')
    plt.legend()
    plt.grid(True)
    
    if verbose:
        plt.savefig(f'{path}/plot_tpr_tnr_intersection.png')
    else:
        plt.show()
    plt.close()
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"TPR at Optimal Threshold: {optimal_tpr:.2f}")
    print(f"TNR at Optimal Threshold: {optimal_tnr:.2f}")
    
    
def find_best_threshold_prev(X_test, y_test, model):
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test,  y_pred_proba)  
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0) # если f1 получила nan Значение
    optimal_threshold = round(thresholds[np.argmax(f1_scores)], 2)
    
    y_pred_binary = [1 if k >= optimal_threshold else 0 for k in y_pred_proba]
    
    return optimal_threshold


def find_feature_importance_catboost(X_train, y_train, cat_features, model, type_, path, save=True):
    pool = Pool(data=X_train, label=y_train, cat_features=cat_features)

    # Проверяем, является ли метод 'ShapValues'
    if type_ == 'ShapValues':
        shap_values = model.get_feature_importance(pool, type='ShapValues')
        
        # ShapValues возвращает матрицу, берем среднее абсолютное значение по всем объектам
        feature_importance = np.abs(shap_values[:, :-1]).mean(axis=0)
    else:
        feature_importance = model.get_feature_importance(pool, type=type_)

    # Создаем DataFrame с важностями
    importance = pd.DataFrame({
        'features': X_train.columns,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)

    # Визуализация
    plt.figure(figsize=(20, 8))
    sns.barplot(y=importance['features'], x=importance['importance'])
    plt.title(f'Feature Importance ({type_})')
    
    if save:
        plt.savefig(f'{path}/feature_importance_{type_}.png')
        plt.close()
    else:
        plt.show()

    return importance


def plot_optimal_proba(y_true, y_pred_prob, path, verbose = True):
        
        # Поиск оптимального порога по F1-score
        precisions, recalls, thresholds = precision_recall_curve(y_true,  y_pred_prob)  
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores, nan=0.0) # если f1 получила nan Значение
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        optimal_f1_score = np.max(f1_scores)

        # Оценка модели с оптимальным порогом
        y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
        accuracy_optimal = accuracy_score(y_true, y_pred_optimal)
        roc_auc_optimal = roc_auc_score(y_true, y_pred_optimal)
        ks_stat_optimal, ks_pvalue_optimal = ks_2samp(y_pred_prob[y_true == 1], y_pred_prob[y_true == 0])
        
        # Построение графика Precision-Recall и отображение оптимального порога
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores[:-1], label='F1-Score')
        plt.xlabel('Threshold')
        plt.ylabel('F1-Score')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
        plt.legend()
        
        
        if verbose:
            plt.savefig(f'{path}/plot_optimal_proba.png')
        else:
            plt.show()
        plt.close()
        
        return round(optimal_threshold, 2)