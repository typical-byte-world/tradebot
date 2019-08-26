import asyncio
import copy
import json
import math
import os
import time
import logging
from time import gmtime, strftime

import binarycom
from neural_network.evaluation import classify
from utils.image_convertor import save_image


async def main():
    loop = asyncio.get_running_loop()
    if not os.path.isdir('images'):
        os.mkdir('images')
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    # logging
    logging.basicConfig(filename=f'logs/{strftime("%Y-%m-%d::%H:%M:%S", gmtime())}.log',
                        filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y::%H:%M:%S')

    with open('configuration.json', mode='r') as configuration_file:
        configuration = json.load(configuration_file)
    websocket = await binarycom.connect(configuration['app_id'])
    print('Авторизируюсь...')
    await binarycom.authorize(websocket, configuration['api_token'])
    print('ОК')
    steps = configuration['steps']
    parameters = copy.deepcopy(configuration['parameters'])
    parameters['amount'] = configuration['base_bet']
    parameters['duration'] = configuration['chart_length'] * configuration['y'] * 60
    parameters['duration_unit'] = 's'
    parameters['currency'] = 'USD'
    total_income = 0
    while True:
        to = math.floor(time.time())
        tick_history = await binarycom.tick_history(websocket, parameters['symbol'],
                                                    to - configuration['chart_length'] * 60, to)
        name = strftime("%Y-%m-%d::%H:%M:%S", gmtime())
        save_image(tick_history['history']['prices'], 'images', f'{name}.png')
        class_ = classify(f'images/{name}.png')

        if class_ == 0:
            await loop.run_in_executor(None, logging.info, f'Image: {name}, result: Неподходящий график')
            print('Неудачный график. Ожидаю...')
            await asyncio.sleep(configuration['wait'] * 60)
            continue
        if class_ == 1:
            print('График на понижение цены.')
            parameters['contract_type'] = 'PUT'
            parameters['barrier'] = - configuration['parameters']['barrier']
            message = 'Понижающийся график'

        else:
            print('График на повышение цены.')
            parameters['contract_type'] = 'CALL'
            parameters['barrier'] = configuration['parameters']['barrier']
            message = 'Повышающийся график'
        await loop.run_in_executor(None, logging.info, f'Image:{name}, result: {message}')

        while True:
            print('Покупаю...')
            balance_before_buy = await binarycom.balance(websocket)
            await binarycom.buy_contract(websocket, parameters)
            await asyncio.sleep(parameters['duration'] + 2)
            balance_after_buy = await binarycom.balance(websocket)
            income = balance_after_buy['balance']['balance'] - balance_before_buy['balance']['balance']
            print(f'Прибыль: {income}')
            total_income = total_income + income
            print(f'Общая прибыль: {total_income}')

            await loop.run_in_executor(None, logging.info,
                               f"Баланс перед покупкой: {balance_before_buy['balance']['balance']},"
                               f" баланс после покупки: {balance_after_buy['balance']['balance']}"
                               f"Доход с последней ставки: {income}, общий доход за текущую авторизацию: {total_income}"
                               f"Степ: {steps}, текущая сумма ставки: {parameters['amount']}"
                               )
            if income < 0:
                await asyncio.sleep(configuration['pause'] * 60)
                if steps > 0:
                    print('Удваиваю ставку...')
                    parameters['amount'] = parameters['amount'] * 2
                    steps = steps - 1
                break
            else:
                await loop.run_in_executor(None, logging.info, f'Начинаю заново. Начальная ставка: {configuration["base_bet"]}')
                print('Устанавливаю базовую ставку...')
                parameters['amount'] = configuration['base_bet']
                steps = configuration['steps']


    await websocket.close()


asyncio.run(main())
