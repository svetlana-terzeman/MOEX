import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class Preproc:
     def clear_data(self, df):
        audit = defaultdict(int)

        # Удаление дублей
        dup_mask = df.duplicated(subset=["date","ticker"])
        audit["duplicates"] = dup_mask.sum()
        if audit["duplicates"]>0:
            print(df[dup_mask])
        df = df[~dup_mask]
        # Пропуски OHLC
        na_mask = df[["open","high","low","close"]].isna().any(axis=1)
        audit["missing_ohlc"] = na_mask.sum()
        if audit["missing_ohlc"]>0:
            print(df[na_mask])
        df = df[~na_mask]
        # Нулевые/отрицательные цены

        bad_price = (df["open"]<=0)|(df["high"]<=0)|(df["low"]<=0)|(df["close"]<=0)
        audit["non_positive_price"] = bad_price.sum()
        if audit["non_positive_price"]>0:
            print(df[bad_price])
        df = df[~bad_price]
        # Логические ошибки свечей
        logic_mask = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        audit["ohlc_logic_error"] = logic_mask.sum()
        if audit["ohlc_logic_error"]>0:
            print(df[logic_mask])
        df = df[~logic_mask]
        # Нулевые объёмы

        zero_vol = df["volume"] <= 0
        audit["zero_volume"] = zero_vol.sum()
        if audit["zero_volume"]>0:
            print(df[zero_vol])
        df = df[~zero_vol]

        return df

     def adjust_splits_by_ticker(self, data, ticker_col='ticker', threshold=0.3,
                            price_cols=None, volume_col='volume', date_col='date'):
        """
        Обнаруживает и корректирует сплиты для каждого тикера отдельно.

        Параметры:
        ----------
        df : pd.DataFrame
            Исходный DataFrame с колонками: тикер, дата, цены, объём.
        ticker_col : str, default='ticker'
            Название колонки с идентификатором тикера.
        threshold : float, default=0.3
            Порог аномального изменения цены за день (30% = 0.3).
        price_cols : list, default=['open','high','low','close']
            Список колонок с ценами для корректировки.
        volume_col : str, default='volume'
            Название колонки с объёмом.
        date_col : str, default='date'
            Название колонки с датой.

        Возвращает:
        ----------
        df_adj : pd.DataFrame
            DataFrame со скорректированными ценами и объёмами.
        """
        df = data.copy()

        if price_cols is None:
            price_cols = ['open', 'high', 'low', 'close']

        # Убедимся, что даты в правильном формате
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Сортируем по тикеру и дате
        df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)
        df['split'] = 'без сплита'
        # Список для хранения обработанных групп
        processed_groups = []

        # Группируем по тикеру и обрабатываем каждую группу
        for ticker, group in df.groupby(ticker_col):
            # Убедимся, что группа отсортирована по дате (уже есть)
            group = group.reset_index(drop=True)
            close = group['close']
            open = group['open']
            # Рассчитываем дневное изменение цены закрытия внутри группы
            pct_change = np.round( open / close.shift(1), 2)  # отношение цена_открытия_сегодня / цена_закрытия_вчера
            group['pct_change'] = pct_change
            # Находим дни-кандидаты на сплит (изменение > порога)
            split_mask = (pct_change > 1 + threshold) | (pct_change < 1 - threshold)
            split_indices = group.index[split_mask]
            split_ratios = pct_change[split_mask]

            if len(split_indices) == 0:
                # Сплитов нет, добавляем группу как есть
                processed_groups.append(group)
                continue

            # Сортируем события от самого позднего к самому раннему
            events = sorted(zip(split_indices, split_ratios), key=lambda x: x[0], reverse=True)

            # Применяем корректировки последовательно
            for idx, split_ratio in events:

                # Маска для строк ДО текущего сплита (в пределах этой группы)
                mask = group.index < idx
                # Корректируем цены (цена вчера * коэфициент, т.к. цену нужно увеличить в split_ratio раз)
                for col in price_cols:
                    group.loc[mask, col] = group.loc[mask, col] * split_ratio
                # Корректируем объём ( объкм вчера / коэфициент,
                # т.к. кол-во акций при сплите при split_ratio> 1 должно уменьшиться
                # а при split_ratio < 1 увеличиться
                group.loc[mask, volume_col] = group.loc[mask, volume_col] / split_ratio

                # Печатаем информацию о событии
                split_date = group.loc[idx, date_col].strftime('%Y-%m-%d')
                split_type = "обратный сплит (консолидация)" if split_ratio > 1 else "прямой сплит"
                group.loc[idx , 'split'] = split_type
                print(f"{ticker} | {split_date}: {split_type}, коэффициент = {split_ratio:.4f}")

            processed_groups.append(group)

        # Объединяем все обработанные группы
        df_adj = pd.concat(processed_groups, ignore_index=True)
        return df_adj

     def preproc(self, df, threshold=0.3):
        df = self.clear_data(df)
        df = self.adjust_splits_by_ticker(df, threshold=threshold)

        return df