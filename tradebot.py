import asyncio
import copy
import json
import math
import time

import binarycom


async def main():
    with open('configuration.json', mode='r') as configuration_file:
        configuration = json.load(configuration_file)
    websocket = await binarycom.connect(configuration['app_id'])
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
        class_ = 1
        # classify the data
        if class_ == 0:
            print('Неудачный график. Ожидаю...')
            await asyncio.sleep(configuration['wait'] * 60)
            continue
        if class_ == 1:
            print('График на понижение цены.')
            parameters['contract_type'] = 'PUT'
            parameters['barrier'] = - 0.459

        else:
            print('График на повышение цены.')
            parameters['contract_type'] = 'CALL'
            parameters['barrier'] = - configuration['barrier']
        while True:
            print('Авторизируюсь...')
            client = await binarycom.authorize(websocket, configuration['api_token'])
            print('Покупаю...')
            response = await binarycom.buy_contract(websocket, parameters)
            income = response['buy']['balance_after'] - client['authorize']['balance']
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

    await websocket.close()


asyncio.run(main())
