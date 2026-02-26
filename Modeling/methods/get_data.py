import pandas as pd
from moexalgo import Market
import logging
from tinkoff.invest import Client, CandleInterval, InstrumentStatus
from pathlib import Path
from settings.constans import TOKEN
from tqdm import tqdm
from datetime import datetime, timedelta
import json
import time


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
logging.getLogger("tinkoff.invest").setLevel(logging.CRITICAL)
logging.getLogger("grpc").setLevel(logging.CRITICAL)

def save_tickers():
    """
    Формирует universe акций первого котировального списка Московской биржи.

    Используется как начальный этап построения датасета.
    Функция обращается к MOEX ISS API, загружает список всех акций и
    оставляет только акции с уровнем листинга 1 (blue chips).

    Возвращает:
        list[str]: список тикеров акций первого эшелона MOEX.
    """

    # Создаем объект для работы с акциями в режиме TQBR
    stocks = Market('stocks')

    # Загружаем данные всех акций
    all_stocks = stocks.tickers()
    df = pd.DataFrame(all_stocks)

    # Фильтруем акции первого котировального списка (уровень листинга = 1)
    first_tier_stocks = df[df['listlevel'] == 1]
    tickers = first_tier_stocks['ticker'].tolist()

    print(f"Найдено акций первого эшелона: {len(tickers)}")
    return tickers


def load_figi_dict():
    """
    Загружает локальный справочник соответствий тикер → FIGI.

    FIGI является обязательным идентификатором для работы с Tinkoff Invest API.
    При первом запуске автоматически создаёт справочник, вызывая create_figi_dict().

    Возвращает:
        dict[str, str]: словарь вида {'GAZP': 'BBG004730ZJ9', ...}
    """
    filepath = Path('data/ticker_figi.json')

    # Создаем родительскую директорию
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Проверяем существование файла
    if not filepath.exists():
        print(f"Файл не найден, создаем новый...")
        if not create_figi_dict():
            return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Успешно загружено {len(data)} записей")
            return data
    except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
        print(f"Ошибка чтения файла: {e}")
        return {}


def create_figi_dict():
    """
    Строит и сохраняет справочник тикер → FIGI из Tinkoff Invest API.

    Загружает все доступные акции и фильтрует только реально торгуемые
    рублевые инструменты.
    """
    try:
        # Получаем список всех акций, доступных для торговли
        with Client(TOKEN) as client:
            instruments = client.instruments.shares(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
            ).instruments

            # Создаем словарь "Тикер -> FIGI" для удобства
            # Отфильтровываем, например, паи и прочее, оставляя только акции
            ticker_to_figi = {
                s.ticker: s.figi
                for s in instruments
                if s.currency == 'rub' and s.buy_available_flag and s.api_trade_available_flag
            }

        filepath = Path('data/ticker_figi.json')
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ticker_to_figi, f, ensure_ascii=False, indent=4)

        print(f"Файл успешно создан: {filepath.absolute()}")
        return True
    except Exception as e:
        print(f"Ошибка при создании файла: {e}")
        return False


def get_data_candles(date_now: datetime, years: int):
    """
    Основной ETL-конвейер загрузки исторических дневных свечей.
    """
    ticker_to_figi = load_figi_dict()
    stocks = save_tickers()
    all_candles = []

    with Client(TOKEN) as client:
        for ticker in tqdm(stocks, desc="Обработка тикеров", unit="тикер"):
            try:
                figi = ticker_to_figi[ticker]
                instrument_info = client.instruments.get_instrument_by(id_type=1, id=figi).instrument

                for period in range(1, (2 * years) + 1):
                    from_date = date_now - timedelta(days=183 * period)
                    to_date = date_now - timedelta(days=183 * (period - 1))

                    candles = list(client.get_all_candles(
                        figi=figi,
                        from_=from_date,
                        to=to_date,
                        interval=CandleInterval.CANDLE_INTERVAL_DAY
                    ))

                    for candle in candles:
                        all_candles.append({
                            'ticker': ticker,
                            'name': instrument_info.name,
                            'datetime': candle.time,
                            'open': float(candle.open.units + candle.open.nano / 1e9),
                            'high': float(candle.high.units + candle.high.nano / 1e9),
                            'low': float(candle.low.units + candle.low.nano / 1e9),
                            'close': float(candle.close.units + candle.close.nano / 1e9),
                            'volume': candle.volume
                        })

                time.sleep(1)
            except Exception as e:
                print(f"Ошибка при получении данных для {ticker}: {e}")

    if all_candles:
        df = pd.DataFrame(all_candles)
        safe_date = date_now.strftime("%Y-%m-%d")
        df.to_parquet(f'prepared_data/candles_data_{safe_date}_{years}years.parquet')
        print("Данные сохранены в parquet")
        return True
    return False


def load_data_candles(date_now: datetime, years: int):
    """
    Универсальная точка доступа к датасету свечей.
    """
    safe_date = date_now.strftime("%Y-%m-%d")
    filepath = Path(f'prepared_data/candles_data_{safe_date}_{years}years.parquet')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        print(f"Файл не найден, создаем новый...")
        if not get_data_candles(date_now, years):
            return pd.DataFrame()

    try:
        df = pd.read_parquet(filepath)
        print(f"Успешно загружено {len(df)} записей")
        return df
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return pd.DataFrame()
