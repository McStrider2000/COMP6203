from mable.cargo_bidding import TradingCompany
from mable.shipping_market import Trade, AuctionLedger, Contract
from mable.cargo_bidding import TradingCompany
from abc import ABC
from typing import List


class AbstractScheduleGenerator(ABC):

    def __init__(self, company: TradingCompany, **kwargs):
        self.company = company
        
    @staticmethod
    def generate_schedule(company: TradingCompany, auction_ledger: AuctionLedger):
        pass

    def pre_inform(self, trades: List[Trade], time: int) -> None:
        pass 
    def inform(self, trades: List[Trade]) -> None:
        pass
    def recieve(self, contract: List[Contract], auction_ledger: AuctionLedger) -> None:
        pass 