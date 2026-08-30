To operate the bot, you need to obtain an API token. **The token must be obtained SEPARATELY FOR EACH ACCOUNT.**

To do this, go to the **"Settings"** tab → **"Security & Limits"**.

Then select **"API Token"**.

In the **"Token name"** field, enter any name you prefer and **MAKE SURE TO CHECK** the **"Read"** and **"Trade"** permissions.

Copy the string from the **"Token"** column and paste it into the `configuration.json` file next to the `api_token` field.

Example:

```json
"api_token": "6ueB7zkVArfD6La"
```

The trading market is specified next to the `symbol` field, which can accept the following values:

```text
"R_10"   for Volatility 10 Index
"R_25"   for Volatility 25 Index
"R_50"   for Volatility 50 Index
"R_75"   for Volatility 75 Index
"R_100"  for Volatility 100 Index

"RDBEAR" for Bear Market Index
"RDBULL" for Bull Market Index
```

Example:

```json
"symbol": "R_75"
```

The barrier is specified next to the `barrier` field.

Example:

```json
"barrier": 0.2
```

The `basis` field can accept either `"stake"` or `"payout"`.

Example:

```json
"basis": "payout"
```
