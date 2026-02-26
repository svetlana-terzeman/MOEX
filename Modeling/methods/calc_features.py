from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas as pd

def calc_features(df):
    # Создаем пустой DataFrame для результатов
    result_dfs = []

    # Обрабатываем каждый тикер отдельно
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        if len(ticker_df) < 50:  # Минимальное количество точек для индикаторов
            continue

        # RSI
        ticker_df['rsi'] = RSIIndicator(ticker_df['close'], window=14).rsi()

        # MACD
        macd = MACD(ticker_df['close'])
        ticker_df['macd'] = macd.macd()
        ticker_df['macd_signal'] = macd.macd_signal()
        ticker_df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(ticker_df['close'], window=20, window_dev=2)
        ticker_df['bb_bbh'] = bb.bollinger_hband()
        ticker_df['bb_bbl'] = bb.bollinger_lband()
        ticker_df['bb_bbm'] = bb.bollinger_mavg()
        ticker_df['bb_bbp'] = bb.bollinger_pband()

        # Простые скользящие средние и волатильность
        ticker_df['ma_5'] = ticker_df['close'].rolling(5).mean()
        ticker_df['ma_20'] = ticker_df['close'].rolling(20).mean()
        ticker_df['daily_return'] = ticker_df['close'].pct_change()
        ticker_df['volatility_5d'] = ticker_df['daily_return'].rolling(5).std()

        # Лаги цен
        for i in [1, 2, 3, 5, 10]:
            ticker_df[f'close_lag_{i}'] = ticker_df['close'].shift(i)

        result_dfs.append(ticker_df)

    # Объединяем все обработанные данные
    df_processed = pd.concat(result_dfs, ignore_index=True)

    return df_processed

def get_feature_descriptions():
    feature_descriptions = {
        # Базовые цены и объем
        'open': 'Цена открытия торгового периода (свечи)',
        'high': 'Максимальная цена в течение торгового периода',
        'low': 'Минимальная цена в течение торгового периода',
        'close': 'Цена закрытия торгового периода',
        'volume': 'Объем торгов (количество акций, торгуемых за период)',

        # RSI индикатор
        'rsi': 'Relative Strength Index (14 периодов) - индекс относительной силы. Показывает перекупленность (>70) и перепроданность (<30)',

        # MACD индикатор
        'macd': 'MACD line - разница между 12-периодной и 26-периодной экспоненциальными скользящими средними',
        'macd_signal': 'Signal line - 9-периодная экспоненциальная скользящая средняя от MACD линии',
        'macd_diff': 'Histogram - разница между MACD и Signal line. Показывает силу тренда',

        # Полосы Боллинджера
        'bb_bbh': 'Bollinger Bands High - верхняя полоса (20 SMA + 2 стандартных отклонения)',
        'bb_bbl': 'Bollinger Bands Low - нижняя полоса (20 SMA - 2 стандартных отклонения)',
        'bb_bbm': 'Bollinger Bands Middle - средняя линия (20-периодная простая скользящая средняя)',
        'bb_bbp': 'Bollinger Bands Percentage - положение цены относительно полос (0-1)',

        # Скользящие средние
        'ma_5': '5-периодная простая скользящая средняя цены закрытия',
        'ma_20': '20-периодная простая скользящая средняя цены закрытия',

        # Доходность и волатильность
        'daily_return': 'Дневная доходность - процентное изменение цены закрытия относительно предыдущего дня',
        'volatility_5d': '5-дневная волатильность - стандартное отклонение дневной доходности за 5 дней',

        # Лаги цен
        'close_lag_1': 'Цена закрытия 1 день назад',
        'close_lag_2': 'Цена закрытия 2 дня назад',
        'close_lag_3': 'Цена закрытия 3 дня назад',
        'close_lag_5': 'Цена закрытия 5 дней назад',
        'close_lag_10': 'Цена закрытия 10 дней назад',

        # Идентификаторы
        'ticker': 'Идентификатор акции (например, SBER, GAZP)',
        'datetime': 'Временная метка торгового периода'
    }

    return feature_descriptions


