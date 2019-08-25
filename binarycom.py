import json
import math

import websockets


def connect(app_id: int):
    return websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}')


async def _do(websocket: websockets.client, request: dict) -> dict:
    await websocket.send(json.dumps(request))
    return json.loads(await websocket.recv())


async def authorize(websocket: websockets.client, api_token: str) -> dict:
    request = {
        "authorize": api_token
    }
    return await _do(websocket, request)


async def tick_history(websocket: websockets.client, symbol: str, from_: int, to: int) -> dict:
    request = {
        "tick_history": symbol,
        "start": from_,
        "end": to
    }
    return await _do(websocket, request)


async def buy_contract(websocket: websockets.client, parameters: dict) -> dict:
    request = {
        "buy": 1,
        "price": math.inf,
        "parameters": parameters
    }
    return await _do(websocket, request)
