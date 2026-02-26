from tinkoff.invest import Client
from tinkoff.invest.schemas import InstrumentIdType, InstrumentType
import pandas as pd
from settings.constans import INSTRUMENT_CONFIGS, TOKEN
import logging
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
logging.getLogger("tinkoff.invest").setLevel(logging.CRITICAL)
logging.getLogger("grpc").setLevel(logging.CRITICAL)




# Основная часть программы
if __name__ == "__main__":
    # Тестируем разные тикеры
    test_tickers = ["ROSN"]
    instrument = 'Акция'
    for ticker in test_tickers:
        # Получаем DataFrame со всеми инструментами
        df = get_instruments_dataframe(ticker, TOKEN)
        columns = ['№', 'Тикер', 'Название', 'Тип', 'Цена', 'Валюта', 'FIGI', 'Лот',
                        'Биржа', 'ISIN', 'Сектор', 'Страна', 'Класс']

        print(df[(
                                                                    (df['Биржа']==INSTRUMENT_CONFIGS[instrument]['stock_market']) &
                                                                    (df['Тип']==INSTRUMENT_CONFIGS[instrument]['type'])
                                                                )][['Тикер', 'Название', 'Тип', 'Цена', 'Валюта', 'Лот']].reset_index(drop=True)
              )


