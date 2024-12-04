from mable.cargo_bidding import TradingCompany
from mable.shipping_market import Trade, AuctionLedger, Contract
from mable.transport_operation import Bid
from abc import ABC
from typing import List


class AbstractBidStrategy(ABC):

    def __init__(self, company: TradingCompany, **kwargs):
        self.company = company
    
    @staticmethod
    def get_bids(schedule):
        pass

    def pre_inform(self, trades: List[Trade], time: int) -> None:
        pass 
    def inform(self, trades: List[Trade]) -> None:
        pass
    def recieve(self, contract: List[Contract], auction_ledger: AuctionLedger) -> None:
        pass 