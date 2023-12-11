import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from ml_model.model import preprocess
import pickle


# Загружаем сохраненную модель с диска
vectorizer, classifier = pickle.load(open("model.pkl", "rb"))

# Создаем бота и диспетчера
bot = Bot(token="6657583932:AAEwcxzEVTt-kwaKLW_-8Rx1PSwcAwlGzkc")
dp = Dispatcher()



# Обрабатываем команду /start
@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    await message.reply("Добро пожаловать! Я бот для анализа текста. Пришли мне сообщение, и я скажу,"
                        " позитивное оно или негативное.")

# Обрабатываем входящие сообщения
@dp.message()
async def classify_message(message: types.Message):
    # Предобрабатываем входящее сообщение
    preprocessed_message = preprocess(message.text)
    # Преобразуем вектор
    message_features = vectorizer.transform([preprocessed_message])
    # Классифицируем сообщение
    prediction = classifier.predict(message_features)[0]
    # Выводим результат
    await message.reply(f"Это {prediction} сообщение!")

async def main():
    await dp.start_polling(bot)

asyncio.run(main())