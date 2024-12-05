from mable.shipping_market import Trade, AuctionLedger, Contract
from mable.transport_operation import Bid
from mable.competition.information import CompanyHeadquarters
from mable.cargo_bidding import TradingCompany
from abc import ABC, abstractmethod
from typing import List, Dict
import logging


class AbstractOpponentSimulation(ABC):

    def __init__(self, company: TradingCompany, **kwargs):
        self.company = company
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_expected_bids() -> Dict[CompanyHeadquarters, List[Bid]]:
        return []

    def pre_inform(self, trades: List[Trade], time: int) -> None:
        pass 
    def inform(self, trades: List[Trade]) -> None:
        pass
    def recieve(self, contract: List[Contract], auction_ledger: AuctionLedger) -> None:
        pass 