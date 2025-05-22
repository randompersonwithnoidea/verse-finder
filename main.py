from environs import Env
from aiogram import Bot, Dispatcher
import asyncio
from handlers import get_router


env = Env()
env.read_env()
bot = Bot(token=env.str("BOT_TOKEN"))
dp = Dispatcher()


dp.include_router(get_router())


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    print("Bot started")
    asyncio.run(main())