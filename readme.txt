Для работы бота необходимо получить API токен. Токен получаеться ОТДЕЛЬНО ДЛЯ КАЖДОГО АККАУНТА.
Что бы это сделать перейдите во вкладку "Settings" -> "Security & Limits".
Далее выберете пункт API Token.
В поле "Token name" введите любое понравившееся вам название и ОБЯЗАТЕЛЬНО ОТМЕТЬТЕ галочки "Read" и "Trade".
Из колонки "Token" скопируйте стркоку и вставте ёё в файл configuration.json напротив поля "api_token".
Пример: "api_token": "6ueB7zkVArfD6La"

Рынок торговли указывается напротив поля "symbol", которое сожет принимать такие значения:
    "R_10" для Volatility 10 Index
    "R_25" для Volatility 25 Index
    "R_50" для Volatility 50 Index
    "R_75" для Volatility 75 Index
    "R_100" для Volatility 10 Index

    "RDBEAR" для Bear Market Index
    "RDBULL" для Bull Market Index

Пример: "symbol": "R_75"

Барьер указывается напротив поля "barrier".
Пример: "barrier": 0.2

Поле "basis" может принимать значения "stake" или "payout".
Пример: "basis": "payout"
