import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr

def get_feature_intervals(df):
    """
    Возвращает словарь с минимальными и максимальными значениями для каждой фичи в DataFrame.
    
    Args:
    df (pd.DataFrame): Входной DataFrame с данными.
    
    Returns:
    dict: Словарь, где ключи - это имена фич, а значения - кортежи (min, max).
    """
    intervals = {}
    
    for column in df.columns:
        if df[column].dtype == object:
            unique_list = list(df[column].unique())
            freq_value  = df[column].mode().iloc[0]
            intervals[column] =  {'unique_list'  : unique_list, 
                                  'freq_value'   : freq_value,
                                  'default_value': 'Other' if ('Other' in unique_list or 'other' in unique_list) else freq_value}
        else:
            min_value  = round(df[column].astype(float).min(), 3)
            max_value  = round(df[column].astype(float).max(), 3)
            mean_value = round(df[column].astype(float).mean(), 3)
            intervals[column] = {'min' : min_value, 'max': max_value, 'mean': mean_value}
        
    return intervals
    
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Вычисляет Mean Absolute Percentage Error (MAPE), игнорируя нулевые значения в y_true.

    Args:
        y_true (list or numpy array): Список истинных значений.
        y_pred (list or numpy array): Список прогнозируемых значений.

    Returns:
        float: MAPE в процентах.
    """
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Определяем ненулевые значения в y_true
    non_zero_mask = y_true != 0

    # Считаем MAPE только для ненулевых значений
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

    return mape

def compare_samples(df1, df2, features_list = 'bets_num'):

    compare_dict = {}

    for feat in features_list:
        
        # ЧИСЛОВЫЕ ФИЧИ
        
        if (df1[feat].nunique() > 2 or df2[feat].nunique() > 2) and (df1[feat].dtypes != object): 
            list1 = df1[feat].describe(percentiles=[i for i in np.arange(0,1,0.05)]).iloc[3:-1].to_list()
            list2 = df2[feat].describe(percentiles=[i for i in np.arange(0,1,0.05)]).iloc[3:-1].to_list()
        
        # БИНАРНЫЕ ФИЧИ 
        else:
            # Получаем все уникальные категории из обоих датафреймов
            all_categories = sorted(set(df1[feat].dropna().unique()) | set(df2[feat].dropna().unique()))
            
            # Создаем словари с нормализованными частотами
            dict1 = df1[feat].value_counts(normalize=True).to_dict()
            dict2 = df2[feat].value_counts(normalize=True).to_dict()
            
            # Создаем списки в одинаковом порядке
            list1 = [dict1.get(cat, 0) for cat in all_categories]
            list2 = [dict2.get(cat, 0) for cat in all_categories]

        if len(list1) > 0 and len(list2) > 0:
            
            max_length = max(len(list1), len(list2))
            list1 += [0] * (max_length - len(list1))
            list2 += [0] * (max_length - len(list2))
            
            # Вычисление Евклидового и Манхэттенского расстояний
            euclidean_distance = euclidean(list1, list2)
            #print(euclidean_distance)
            manhattan_distance = cityblock(list1, list2)
    
            # Нормализация расстояний к диапазону от 0 до 1
            max_euclidean_distance = np.sqrt(np.sum(np.square(np.maximum(list1, list2))))  # Максимально возможное Евклидово расстояние
            euclidean_similarity = (1 - (euclidean_distance / max_euclidean_distance))*100
            
            max_manhattan_distance = np.sum(np.abs(np.maximum(list1, list2)))  # Максимально возможное Манхэттенское расстояние
            manhattan_similarity = (1 - (manhattan_distance / max_manhattan_distance))*100
    
            # Вычисление коэффициента корреляции Пирсона
            pearson_corr, _ = pearsonr(list1, list2)
    
            
    
            mape_list = []
            for idx in range(0,len(list1),1):
                
                mape = mean_absolute_percentage_error(list1[idx], list2[idx])
                
                if not pd.isna(mape):
                    mape_list.append(mape)
            
    
            compare_dict[feat] = {
                'euclidean':euclidean_similarity,
                'manhattan':manhattan_similarity,
                #'pearson':pearson_corr,
                'mape_avg':np.mean(mape_list),
            }
        else:
            compare_dict[feat] = {
                 'euclidean':np.nan,
                'manhattan':np.nan,
             #   'pearson':np.nan,
                'mape_avg':np.nan,
            }

    compare_dict = pd.DataFrame(compare_dict).T
    return compare_dict
    
# функция удаления фичей с низкой дисперсией
def remove_low_variance_features(df, features, threshold=2):
    variance = df[features].var()
    display(variance)
    # Определяем признаки с дисперсией ниже порога
    low_variance_features = variance[variance < threshold].round(3).to_dict()
    return low_variance_features

def const_feature(df: pd.DataFrame, features: list, threshold: float = 0.8) -> list:
    """
    Удаляет признаки, в которых одно значение занимает долю больше threshold.

    :param df: DataFrame с данными.
    :param features: Список фичей для проверки.
    :param threshold: Порог, выше которого фича считается квази-константной.
    :return: Список оставшихся фичей.
    """
    remaining_features = {}
    for feature in features:
        top_freq = df[feature].value_counts(normalize=True, dropna=False).head(1)
        if round(top_freq.values[0], 3) >= threshold:
            remaining_features[feature] = f'Мажоритарное значение {top_freq.index[0]}, доля значения - {round(top_freq.values[0], 3)}'
    return remaining_features

# функция для нахождения фичей, где пропусков больше чем N
def nan_values(df, features_lst, N):
    """
        Функция удаления фичей с пропусками, с долей больше чем N
        Параметры:
            df: pd.Dataframe
                Датафрейм
            features_lst: list
                Список фичей
            N: float
                Доля пропусков в выборке 
    """
    features_list = features_lst.copy()
    # N - параметр отвечающий за допустимый процент пропусков в данных
    # смотрим процент пропусков в фичах
    drop_features = (df[features_list].isna().mean() <= N) \
            .astype(int).reset_index().rename({0:'value', 'index':'feature'}, axis = 1).to_dict('records')
    
    # удаляем столбцы в которых много пропусков
    nan_dict = {}
    for current in drop_features:
        if current['value'] == 0:
            nan_dict[current['feature']] = f'Доля пропусков по фиче: {df[current["feature"]].isna().mean().round(5)}'
    return nan_dict

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    
    ''' Функция заполнения пропусков в данных '''
    df_filled = df.copy()
    
    for col in df_filled.columns:
        unique_types = df_filled[col].dropna().map(type).unique()
        
        if set(unique_types) == {bool} or set(unique_types) == {bool, type(None)}:
            # Принудительно преобразуем к bool и заполняем False
            df_filled[col] = df_filled[col].astype('boolean').fillna(False)
        elif pd.api.types.is_numeric_dtype(df_filled[col]):  
            df_filled[col].fillna(0, inplace=True)
        elif df_filled[col].dtype == 'object':
            df_filled[col] = df_filled[col].replace('', 'n/d')
            df_filled[col].fillna('n/d', inplace=True)
    
    return df_filled