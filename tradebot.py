import asyncio
import json
import math
import time

import binarycom


class TradeBot:
    def __init__(self, configuration_: dict):
        self.api_token = configuration_['api_token']
        self.websocket = None
        self.chart_length = configuration_['chart_length']
        self.wait = configuration_['wait']
        self.symbol = configuration_['symbol']

    @staticmethod
    async def new(configuration_):
        self = TradeBot(configuration_)
        self.websocket = await binarycom.connect(configuration_['app_id'])
        return self

    async def authorize(self):
        return await binarycom.authorize(self.websocket, self.api_token)

    async def tick_history(self):
        to = math.floor(time.time())
        return await binarycom.tick_history(self.websocket, self.symbol, to - self.chart_length * 60, to)

    async def buy_contract(self) -> dict:
        await self.authorize()
        return await binarycom.buy_contract(self.websocket, {})


async def main():
    with open('configuration.json', mode='r') as configuration_file:
        configuration = json.load(configuration_file)
    trade_bot = await TradeBot.new(configuration)

    # загрузить данные
    # проверить
    # если плохо - ждём wait
    # иначе вычисляем duration = chart_length * y
    # делаем ставку
    # если проиграли - ждем pause и удваиваем ставку
    #
    while True:
        tick_history = trade_bot.tick_history()
        class_ = 1
        # classify the data
        if class_ == 0:
            await asyncio.sleep(trade_bot.wait * 60)
            continue
        if class_ == 1:
            # DOWN
            continue
        else:
            print('')

        # UP
        trade_bot.buy_contract()


asyncio.run(main())
