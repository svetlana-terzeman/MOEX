import pickle
import pandas as pd
import json
import numpy as np
from datetime import datetime
from methods.load_data import LoadData
from methods.preproc import Preproc
from methods.feature_endineering import FeatureEngineering
from catboost import CatBoostClassifier
from apscheduler.schedulers.blocking import BlockingScheduler
import pytz
import warnings
warnings.filterwarnings("ignore")

def start():
    print('Начало загрузки данных о модели')
    # Загрузка данных
    path = 'models'
    model = CatBoostClassifier()
    model.load_model(f'{path}/catboost_model.cbm')

    with open(f'{path}/tickers.pkl', 'rb') as f:
        tickers = pickle.load(f)

    with open(f'{path}/history_dataset.pickle', 'rb') as f:
        df_h = pickle.load(f)

    with open(f"{path}/features_intervals.json", "r", encoding="utf-8") as f:
        features_intervals = json.load(f)

    with open(f"{path}/threshold.json", "r", encoding="utf-8") as f:
        threshold = json.load(f)
    print('Данные модели успешно загруженны')
    features = model.feature_names_
    cat_features = []

    # Получаем данные за сегодня после закрытия рынка
    print('Страрт загрузки данныс с MOEX')
    start = df_h.date.max()
    load_cls = LoadData()
    df, skipped = load_cls.load_moex_universe(tickers, start_date=start)
    print('Данные загруженны')

    print(f'df_h: {df_h.shape}, df: {df.shape}')
    df = pd.concat([df_h, df])

    print('Старт скоринга')
    preproc_cls = Preproc()
    df = preproc_cls.preproc(df, threshold=0.3)

    df = df.sort_values(by =['ticker', 'date'])
    df = df.groupby(['ticker']).tail(100)


    feature_cls = FeatureEngineering()
    df = feature_cls.feature_eng(df, features_intervals, cat_features)

    df['probability'] = model.predict_proba(df[features])[:, 1]  # Predict probabilities for the positive class
    df['predict'] = (df['probability'] >= threshold).astype(int) #

    df = df.sort_values(by=['ticker', 'date'])
    result = df.groupby(['ticker']).tail(1)

    result = result[['ticker', 'predict', 'probability']]
    result['trend'] = np.where(result['predict']==1, 'UP', 'DOWN')
    result['probability'] = np.where(result['predict']==1, result['probability'], 1-result['probability'])
    print('Скоринг выполнен успешно')
    # Устанавливаем ticker как индекс, выбираем нужные колонки и преобразуем
    result_dict = result.set_index('ticker')[['trend', 'probability']].to_dict(orient='index')

    with open(f'{path}/history_dataset.pickle', 'wb') as f:
        pickle.dump(df_h, f)

    with open("result/signals.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print('Результаты сохранены успешно')

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone=pytz.timezone('Europe/Moscow'))
    scheduler.add_job(start, 'cron', hour=23, minute=00)
    scheduler.start()
    # start()
