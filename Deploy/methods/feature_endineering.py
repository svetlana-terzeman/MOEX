import pandas as pd
import numpy  as np


class FeatureEngineering:
    def calc_features(self, df):

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Доходность — это процентное изменение цены акции за выбранный период времени.
        # Она показывает, насколько выросла или упала цена по сравнению с прошлым значением.

        # Экономический смысл:
        # Если доходность положительная — акция растёт.
        # Если отрицательная — акция падает.
        # Чем больше модуль доходности, тем сильнее движение цены.

        df["ret_1"] = df.groupby("ticker")["close"].pct_change(1)  # дневное изменение цены
        df["ret_3"] = df.groupby("ticker")["close"].pct_change(1)  # изменение цены за три дня
        df["ret_5"] = df.groupby("ticker")["close"].pct_change(5)  # изменение за торговую неделю
        df["ret_20"] = df.groupby("ticker")["close"].pct_change(20)  # изменение за торговый месяц
        # Волатильность показывает, насколько нестабильно ведёт себя цена акции.
        # Чем выше волатильность — тем выше риск и тем более резкие движения цены происходят.
        #
        # Экономический смысл:
        # Высокая волатильность означает нервный, неустойчивый рынок.
        # Низкая — спокойное, стабильное движение.

        df["vol3"] = df.groupby("ticker")["ret_1"].rolling(3).std().reset_index(0, drop=True)  # волатильность за 3 дня
        df["vol_5"] = df.groupby("ticker")["ret_1"].rolling(5).std().reset_index(0,
                                                                                 drop=True)  # волатильность за 5 дней
        df["vol_15"] = df.groupby("ticker")["ret_1"].rolling(15).std().reset_index(0,
                                                                                   drop=True)  # волатильность за 15 дней
        df["vol_30"] = df.groupby("ticker")["ret_1"].rolling(30).std().reset_index(0,
                                                                                   drop=True)  # волатильность за 30 дней
        df["vol_45"] = df.groupby("ticker")["ret_1"].rolling(45).std().reset_index(0,
                                                                                   drop=True)  # волатильность за 45 дней
        df["vol_90"] = df.groupby("ticker")["ret_1"].rolling(90).std().reset_index(0,
                                                                                   drop=True)  # волатильность за 90 дней
        # Моментум показывает, насколько сильно и в каком направлении двигалась цена акции
        # за выбранный период времени.
        #
        # Экономический смысл:
        # Положительный моментум означает, что акция находится в фазе роста.
        # Отрицательный — что акция находится в фазе падения.

        df["mom_3"] = df.groupby("ticker")["close"].pct_change(3)  # сила движения за 3 дня
        df["mom_5"] = df.groupby("ticker")["close"].pct_change(5)  # сила движения за 5 дней
        df["mom_15"] = df.groupby("ticker")["close"].pct_change(15)  # сила движения за 5 дней
        df["mom_20"] = df.groupby("ticker")["close"].pct_change(20)  # сила движения за 20 дней
        # Скользящие средние показывают сглаженное направление движения цены акции.
        # Они позволяют отделить устойчивый тренд от случайных рыночных колебаний.
        #
        # Экономический смысл:
        # Если короткая средняя выше длинной — тренд восходящий.
        # Если ниже — нисходящий.

        df["sma_3"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(3).mean())  # средняя цена за 3 дня
        df["sma_5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())  # средняя цена за 5 дней
        df["sma_10"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(10).mean())  # средняя цена за 5 дней
        df["sma_20"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(20).mean())  # средняя цена за 20 дней

        df["sma_ratio"] = df["sma_5"] / df["sma_20"] - 1  # относительное положение краткой и длинной средних

        # Объём торгов показывает активность участников рынка.
        # Он отражает, насколько активно инвесторы покупают и продают акцию.
        #
        # Экономический смысл:
        # Рост объёма подтверждает силу движения цены.
        # Падение объёма — признак ослабления интереса к акции.

        df["vol_mean_5"] = df.groupby("ticker")["volume"].transform(
            lambda x: x.rolling(5).mean())  # средний объём за 5 дней
        df["vol_mean_15"] = df.groupby("ticker")["volume"].transform(
            lambda x: x.rolling(15).mean())  # средний объём за 15 дней
        df["vol_mean_45"] = df.groupby("ticker")["volume"].transform(
            lambda x: x.rolling(45).mean())  # средний объём за 45 дней
        df["vol_ratio"] = df["volume"] / df["vol_mean_5"]  # относительный объём

        # Свечной диапазон показывает внутридневную амплитуду колебаний цены.
        # Он отражает уровень рыночной неопределённости и борьбу покупателей и продавцов.

        df["hl_range"] = (df["high"] - df["low"]) / df["close"]  # относительный дневной диапазон

        # Временные признаки отражают календарную структуру торговых сессий.
        # Рынок ведёт себя по-разному в разные дни недели и месяцы года из-за поведения инвесторов,
        # налоговых периодов, отчётностей и ребалансировок фондов.

        # Базовые
        df["dow"] = df["date"].dt.weekday  # день недели (0 = пн, 6 = вс)
        df["month"] = df["date"].dt.month  # номер месяца (1-12)
        df["week"] = df["date"].dt.isocalendar().week.astype(int)  # неделя года (1-52)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

        # Дополнительные признаки
        df["day_of_month"] = df["date"].dt.day  # число месяца (1-31)
        df["day_of_year"] = df["date"].dt.dayofyear  # день года (1-366)
        df["quarter"] = df["date"].dt.quarter  # квартал (1-4)
        df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
        df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
        df["is_weekend"] = (df["dow"] >= 5).astype(int)  # суббота/воскресенье = 1

        # Для учёта цикличности (полезно для CatBoost, но можно и так)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df


    def replace_features_intervals(self, df, features_intervals, cat_features):
        # пришла пустота на инференс ты меняешь на наиболее вероятное/популярное (для категориальной)  для числовой на среднее
        # если числовая пришла выше границы, то замечанием на верзнюю границу если нижнняя на нижнюю

        # если пришло значение но его нет в словаре заменяем на other
        # как быть если не обучалась на other но пришло на инфуренс категориальное значение которго не было на обучении
        # заменять на наиболее вероятное

        #  замена новых значений в test
        for feature in features_intervals.keys():
            if feature in cat_features:
                df.loc[df[feature].isna(), feature] = features_intervals[feature]['freq_value']
                df.loc[~ f[feature].isin(features_intervals[feature]['unique_list']), feature] = \
                features_intervals[feature]['default_value']

            else:
                df.loc[df[feature].isna(), feature] = features_intervals[feature]['mean']
                df.loc[df[feature] > features_intervals[feature]['max'], feature] = features_intervals[feature]['max']
                df.loc[df[feature] < features_intervals[feature]['min'], feature] = features_intervals[feature]['min']
        return df

    def feature_eng(self, df, features_intervals, cat_features):
        df = self.calc_features(df)
        df = self.replace_features_intervals(df, features_intervals, cat_features)

        return df