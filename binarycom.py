import json

import websockets


async def connect(app_id: int):
    return await websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}')


async def _do(websocket, request: dict) -> dict:
    await websocket.send(json.dumps(request))
    return json.loads(await websocket.recv())


async def authorize(websocket, api_token: str) -> dict:
    request = {
        'authorize': api_token
    }
    return await _do(websocket, request)


async def tick_history(websocket, symbol: str, from_: int, to: int) -> dict:
    request = {
        'ticks_history': symbol,
        'start': from_,
        'end': to
    }
    return await _do(websocket, request)


async def buy_contract(websocket, parameters: dict) -> dict:
    request = {
        'buy': 1,
        'price': 9999,
        'parameters': parameters
    }
    return await _do(websocket, request)


async def balance(websocket):
    request = {
        'balance': 1
    }
    return await _do(websocket, request)
