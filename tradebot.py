import asyncio
import json

import binarycom


class TradeBot:
    def __init__(self, configuration: dict):
        self.websocket = binarycom.connect(configuration["app_id"])

    async def buy_contract(self) -> dict:
        return await binarycom.buy_contract(self.websocket, {})


async def main():
    with open('configuration.json', mode='r') as configuration_file:
        configuration = json.load(configuration_file)
    trade_bot = TradeBot(configuration)
    while True:
        await asyncio.sleep(10)


asyncio.run(main())
