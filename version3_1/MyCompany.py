from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.transport_operation import Bid
from mable.shipping_market import Trade, AuctionLedger
from mable.shipping_market import Contract
from mable.competition.information import CompanyHeadquarters
from BidStrategy import AbstractBidStrategy
from typing import List
import logging

from ScheduleGenerator import AbstractScheduleGenerator
from OpponentSimulation import AbstractOpponentSimulation

class MyCompany(TradingCompany):
    logger: logging.Logger
    headerquarters: CompanyHeadquarters
    fleet: List[VesselWithEngine]

    bid_strategy: AbstractBidStrategy
    opponent_simulation: AbstractOpponentSimulation
    schedule_generator: AbstractScheduleGenerator

    def __init__(self, fleet: List[VesselWithEngine], name: str):
        super().__init__(fleet, name)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create 3 key parts
        self.schedule_generator = AbstractScheduleGenerator(company=self)
        self.opponent_simulation = AbstractOpponentSimulation(company=self)
        self.bid_strategy = AbstractBidStrategy(company=self)

    def pre_inform(self, trades: List[Trade], time: int):
        self.schedule_generator.pre_inform(trades, time)
        self.opponent_simulation.pre_inform(trades, time)
        self.bid_strategy.pre_inform(trades, time)
        pass 

    def inform(self, trades: List[Trade]) -> List[Bid]:
        self.schedule_generator.inform(trades)
        self.opponent_simulation.inform(trades)
        self.bid_strategy.inform(trades)
        return self.bid_strategy.get_bids()
    
    def recieve(self, contract: List[Contract], auction_ledger: AuctionLedger):
        self.schedule_generator.recieve(contract, auction_ledger)
        self.opponent_simulation.recieve(contract, auction_ledger)
        self.bid_strategy.recieve(contract, auction_ledger)
        pass 