import json
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

TOKEN = "8798560559:AAGtuOvNMpo6_Gc95UMPwGkzCQ1Wvq18Xns"

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


def load_signals():
    with open("result/signals.json", "r", encoding="utf-8") as f:
        return json.load(f)


@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    text = """
            AI бот прогнозирования акций MOEX
            
            Команды:
            /top
            /stock SBER
            """
    await message.answer(text)


@dp.message_handler(commands=["top"])
async def top_signals(message: types.Message):
    data = load_signals()
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index().rename(columns={'index': 'ticker'})
    df = df.sort_values(['probability'], ascending=False).head(5)

    text = "Топ сигналы:\n\n"

    for _, row in df.iterrows():
        text += (
            f"{row['ticker']}\n"
            f"Тренд: {row['trend']}\n"
            f"Вероятность: {round(row['probability'] * 100)}%\n\n"
        )

    await message.answer(text)


@dp.message_handler(commands=["stock"])
async def stock(message: types.Message):
    data = load_signals()

    try:
        ticker = message.text.split()[1].upper()
    except:
        await message.answer("Пример: /stock SBER")
        return

    if ticker not in data:
        await message.answer("Нет данных")
        return

    info = data[ticker]

    text = (
        f"{ticker}\n\n"
        f"Тренд: {info['trend']}\n"
        f"Вероятность: {round(info['probability']*100)}%\n\n"
    )

    await message.answer(text)


if __name__ == "__main__":
    executor.start_polling(dp)