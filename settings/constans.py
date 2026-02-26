import os
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv('TINKOFF_TOKEN', None)

INSTRUMENT_CONFIGS = {
                        'Акция': {
                                        'type': 'share',
                                        'stock_market': 'moex_mrng_evng_e_wknd_dlr'
                        },
                        'Облигация' : {
                                        'type': 'bond',
                                        'stock_market' : 'moex_bonds'
                        },
                        'Фьючерс': {
                                        'type': 'futures',
                                        'stock_market': 'forts_futures_weekend'
                        }
                    }