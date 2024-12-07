from mable.cargo_bidding import TradingCompany
from mable.shipping_market import Trade, AuctionLedger, Contract
from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal
from abc import ABC, abstractmethod
from typing import List
import logging


class AbstractScheduleGenerator(ABC):
    """
    Abstract class for generating schedules.
    """

    def __init__(self, company: TradingCompany, **kwargs):
        self.company = company
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def generate(self, company: TradingCompany, trades: List[Trade]) -> ScheduleProposal:
        pass

    def pre_inform(self, trades: List[Trade], time: int) -> None:
        pass 
    def inform(self, trades: List[Trade]) -> None:
        pass
    def receive(self, contract: List[Contract], auction_ledger: AuctionLedger) -> None:
        pass 