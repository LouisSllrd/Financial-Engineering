from datetime import date
from gs_quant.data import DataContext
from gs_quant.markets.securities import SecurityMaster, AssetIdentifier, ExchangeCode
import gs_quant.timeseries as ts

data_ctx = DataContext(start=date(2018, 1, 1), end=date(2018, 12, 31))      # Create a data context covering 2018
spx = SecurityMaster.get_asset('SPX', AssetIdentifier.TICKER, exchange_code=ExchangeCode.NYSE)               # Lookup S&P 500 Index via Security Master

with data_ctx:                                                              # Use the data context we setup
    vol = ts.implied_volatility(spx, '1m', ts.VolReference.DELTA_CALL, 25)  # Get 25 delta call implied volatility

vol.tail()