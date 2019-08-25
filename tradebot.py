import asyncio
import copy
import json
import logging
import math
import time

from neural_network.evaluation import classify
from utils.image_convertor import save_image
import binarycom


async def main():
    logging.basicConfig(level=logging.INFO)
    with open('configuration.json', mode='r') as configuration_file:
        configuration = json.load(configuration_file)
    websocket = binarycom.connect(configuration['app_id'])
    steps = configuration['steps']
    parameters = copy.deepcopy(configuration['parameters'])
    parameters['amount'] = configuration['base_bet']
    parameters['duration'] = configuration['chart_length'] * configuration['y'] * 60
    parameters['duration_unit'] = 's'
    parameters['currency'] = 'USD'
    while True:
        to = math.floor(time.time())
        tick_history = await binarycom.tick_history(websocket, parameters['symbol'],
                                                    to - configuration['chart_length'] * 60, to)
        img_array = tick_history['history']['prices']

        # name
        name = time.time()
        save_image(img_array,'tmp', f'{name}.png')
        class_ = classify(f'utils/{name}')

        if class_ == 0:
            logging.info('Неудачный график. Ожидаю...')
            await asyncio.sleep(configuration['wait'] * 60)
            continue
        if class_ == 1:
            logging.info('График на понижение цены.')
            parameters['contract_type'] = 'PUT'
            parameters['barrier'] = - configuration['barrier']

        else:
            logging.info('График на повышение цены.')
            parameters['contract_type'] = 'CALL'
            parameters['barrier'] = configuration['barrier']
        while True:
            logging.info('Авторизируюсь...')
            client = await binarycom.authorize(websocket, configuration['api_token'])
            logging.info('Покупаю...')
            response = await binarycom.buy_contract(websocket, parameters)
            income = client['authorize']['balance'] - response['buy']['balance_after']
            logging.info(f'Прибыль: {income}')
            if income < 0:
                await asyncio.sleep(configuration['pause'])
                if steps > 0:
                    parameters['amount'] = parameters['amount'] * 2
                    steps = steps - 1
                break
            else:
                parameters['amount'] = configuration['base_bet']

    await websocket.close()


asyncio.run(main())
