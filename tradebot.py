import asyncio
import copy
import json
import math
import os
import time

import binarycom
from neural_network.evaluation import classify
from utils.image_convertor import save_image


async def main():
    with open('configuration.json', mode='r') as configuration_file:
        configuration = json.load(configuration_file)
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    websocket = await binarycom.connect(configuration['app_id'])
    await binarycom.authorize(websocket, configuration['api_token'])
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
        name = time.time()
        save_image(tick_history['history']['prices'], 'tmp', f'{name}.png')
        class_ = classify(f'tmp/{name}.png')

        if class_ == 0:
            print('Неудачный график. Ожидаю...')
            await asyncio.sleep(configuration['wait'] * 60)
            continue
        if class_ == 1:
            print('График на понижение цены.')
            parameters['contract_type'] = 'PUT'
            parameters['barrier'] = - configuration['parameters']['barrier']

        else:
            print('График на повышение цены.')
            parameters['contract_type'] = 'CALL'
            parameters['barrier'] = configuration['parameters']['barrier']
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
            if income < 0:
                await asyncio.sleep(configuration['pause'] * 60)
                if steps > 0:
                    print('Удваиваю ставку...')
                    parameters['amount'] = parameters['amount'] * 2
                    steps = steps - 1
                break
            else:
                parameters['amount'] = configuration['base_bet']
                steps = configuration['steps']

    await websocket.close()


asyncio.run(main())
