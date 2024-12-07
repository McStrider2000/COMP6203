from mable.cargo_bidding import TradingCompany

class MostBasicCompany(TradingCompany):
    """
    A company that does nothing. It does not bid on any trades.
    Useful for testing purposes.
    """
    def inform(self, trades, *args, **kwargs):
        return []