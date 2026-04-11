import requests
import pandas as pd
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class LoadData:
    def __init__(self):
        # Создаём объект сессии для выполнения HTTP-запросов
        self.SESSION = requests.Session()

        # Настраиваем стратегию повторных попыток на случай временных сбоев
        retries = Retry(
            total=10,  # Максимальное количество попыток (включая первую)
            backoff_factor=0.5,  # Задержка = {backoff_factor} * (2 ** (попытка - 1))
            status_forcelist=[500, 502, 503, 504],  # Список HTTP-статусов, при которых нужно повторить запрос
            allowed_methods=["GET"]  # Разрешаем повтор только для идемпотентных методов (GET).
        )

        # Монтируем HTTPAdapter к сессии для протокола HTTPS.
        # Адаптер отвечает за фактическую отправку запросов и применяет
        # настроенную стратегию повторных попыток ко всем HTTPS-запросам через эту сессию.
        self. SESSION.mount("https://", HTTPAdapter(max_retries=retries))
    def load_moex_history_one(self, ticker, start_date="2014-01-01", end_date=None, board="TQBR"):
        """
        Загружает полную историческую дневную OHLCV-серию по одной акции с официального API MOEX.

        Функция автоматически обходит ограничение MOEX на размер ответа (100 строк на запрос)
        за счёт постраничной загрузки данных (pagination) и объединяет все страницы
        в единую временную серию.

        Параметры
        ----------
        ticker : str
            Биржевой тикер акции (например: 'SBER', 'GAZP', 'LKOH').

        start_date : str, default='2014-01-01'
            Начальная дата исторической выборки в формате 'YYYY-MM-DD'.

        end_date : str | None, default=None
            Конечная дата выборки. Если None — данные загружаются по последний торговый день.

        board : str, default='TQBR'
            Основной режим торгов MOEX. 'TQBR' используется для получения наиболее ликвидных
            и корректных рыночных котировок.
        """

        # Исторический эндпоинт MOEX для рынка акций
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"

        all_rows = []   # Сюда будут накапливаться все страницы истории
        start = 0       # Смещение для постраничной загрузки (pagination)

        # MOEX выдаёт не более 100 строк за один запрос,
        # поэтому история загружается в цикле с постепенным смещением "start"
        while True:
            r = self.SESSION.get(
                url,
                params={
                    "from": start_date,     # дата начала выборки
                    "till": end_date,       # дата окончания выборки
                    "start": start,         # смещение страницы
                    "iss.meta": "off",      # отключаем служебную мета-информацию
                },
                timeout=50
            )

            # Если сервер вернул ошибку — прекращаем загрузку для данного тикера
            if r.status_code != 200:
                break

            js = r.json()
            cols = js["history"]["columns"]
            data = js["history"]["data"]

            # Если сервер не вернул данных — история закончилась
            if not data:
                break

            # Преобразуем текущую страницу в DataFrame и добавляем в общий список
            all_rows.append(pd.DataFrame(data, columns=cols))

            # MOEX всегда отдаёт максимум 100 строк.
            # Если получено меньше — это последняя страница истории.
            if len(data) < 100:
                break

            # Смещаемся на следующую страницу
            start += 100

            # Небольшая задержка между запросами — обязательна,
            # чтобы избежать серверных ограничений и обрывов соединения
            time.sleep(0.2)

        # Если MOEX не вернул ни одной страницы — возвращаем пустой DataFrame
        if not all_rows:
            return pd.DataFrame()

        # Объединяем все страницы истории в единую временную серию
        df = pd.concat(all_rows)

        # Оставляем только финансово значимые поля OHLCV
        df = df[["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = [c.lower() for c in df.columns]

        # Приводим дату к формату datetime и сортируем временной ряд
        df["tradedate"] = pd.to_datetime(df["tradedate"])
        df = df.sort_values("tradedate").set_index("tradedate")

        # Добавляем тикер как признак для формирования panel-датасета
        df["ticker"] = ticker

        return df


    def load_moex_universe(
        self,
        tickers: list[str],
        start_date: str = "2014-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Загружает исторические котировки для всех акций торгового универсума MOEX
        и формирует единый panel-датасет для последующего машинного обучения.

        Каждая акция загружается отдельно через load_moex_history_one(), после чего
        все временные ряды объединяются в одну таблицу формата:

        [date, open, high, low, close, volume, ticker]

        Такой формат называется panel data и является стандартом для финансового ML.


        Параметры
        ----------
        tickers : list[str]
            Список биржевых тикеров акций, формирующих торговый универсум
            (например: ['SBER', 'GAZP', 'LKOH', ...]).

        start_date : str, default='2014-01-01'
            Начальная дата выборки в формате 'YYYY-MM-DD'.
            Определяет начало исторического периода, используемого для обучения модели.

        end_date : str | None, default=None
            Конечная дата выборки в формате 'YYYY-MM-DD'.
            Если None — данные загружаются по последнюю доступную торговую сессию.

        Возвращает
        ----------
        pd.DataFrame
            Единый panel-датасет формата:
            [date, open, high, low, close, volume, ticker],
            где каждая строка соответствует одной акции в один торговый день.
        """

        frames = []   # сюда будут собираться DataFrame'ы с котировками по каждой акции
        bad = []      # список тикеров, по которым не удалось получить корректные данные

        # Проходим по всем акциям торгового универсума
        for t in tqdm(tickers, desc="Loading MOEX universe"):
            try:
                # Загружаем исторические данные по одной акции
                dft = self.load_moex_history_one(
                            t,                     # тикер акции (например: 'SBER', 'GAZP', 'LKOH')
                            start_date=start_date, # дата начала исторической выборки
                            end_date=end_date,     # дата окончания (None = по последний торговый день)
                            board = "TQBR"         # режим торгов
                        )
                time.sleep(0.3)
                # Иногда MOEX возвращает пустой набор (делистинг, технические бумаги и т.п.)
                # Такие акции исключаем из датасета, чтобы не загрязнять модель шумом
                if dft.empty:
                    print('bad')
                    bad.append(t)
                    continue

                frames.append(dft)

            # В случае любой сетевой или серверной ошибки просто исключаем тикер
            # и продолжаем сбор данных по остальным акциям
            except Exception:
                bad.append(t)

        # Объединяем котировки всех акций в один большой panel dataset
        panel = pd.concat(frames, axis=0).reset_index().rename(columns={"tradedate": "date"})

        # Итоговый формат данных:
        # date | open | high | low | close | volume | ticker
        # где каждая строка — один торговый день по одной акции
        return panel, bad
