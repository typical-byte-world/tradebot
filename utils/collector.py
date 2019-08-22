#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import math
import os
import random
import string
import time

import websockets


async def get_tick_history(uri: str, symbol: str, from_: int, to: int) -> dict:
    async with websockets.connect(uri) as websocket:
        request = {
            "ticks_history": symbol,
            "start": from_,
            "end": to
        }
        await websocket.send(json.dumps(request))
        return json.loads(await websocket.recv())['history']


def get_random_string(length=16) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


def convert_to_csv(tick_history: dict, file_name: str):
    with open(file_name, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'price'])
        for time_, price in zip(tick_history['times'], tick_history['prices']):
            writer.writerow([time_, price])


async def get_tick_history_and_convert_to_csv(uri: str, symbol: str, from_: int, to: int, directory: str):
    while True:
        file_name = os.path.join(directory, f'{get_random_string()}.csv')
        if not os.path.isfile(file_name):
            break
    convert_to_csv(
        await get_tick_history(uri, symbol, from_, to),
        file_name
    )


async def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--app_id', type=int, required=True, help='application ID', metavar='id')
    argument_parser.add_argument('--coverage', type=int, required=True, help='csv coverage in minutes',
                                 metavar='minutes')
    argument_parser.add_argument('--number', type=int, required=True, help='number of csv', metavar='number')
    argument_parser.add_argument('--directory', default='', type=str, help='change directory', metavar='directory')
    arguments = argument_parser.parse_args()
    uri = f'wss://ws.binaryws.com/websockets/v3?app_id={arguments.app_id}'
    to = math.floor(time.time())
    if not os.path.isdir(arguments.directory):
        os.mkdir(arguments.directory)
    for i in range(4):
        offset = i * (arguments.number // 4) * arguments.coverage * 60
        await asyncio.gather(
            *[
                get_tick_history_and_convert_to_csv(
                    uri,
                    'R_10',
                    to - offset - (j + 1) * arguments.coverage * 60,
                    to - offset - j * arguments.coverage * 60,
                    arguments.directory
                ) for j in range(arguments.number // 4)
            ]
        )


asyncio.run(main())
