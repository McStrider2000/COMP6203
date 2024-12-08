from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.transport_operation import Bid
from mable.shipping_market import Trade, AuctionLedger
from mable.shipping_market import Contract
from mable.competition.information import CompanyHeadquarters
from BidStrategy import AbstractBidStrategy
from typing import List
import logging

from ScheduleGenerator.AbstractScheduleGenerator import AbstractScheduleGenerator
from ScheduleGenerator.BruteScheduleGenerator import BruteScheduleGenerator
from OpponentSimulation.AbstractOpponentSimulation import AbstractOpponentSimulation
from OpponentSimulation.DumbSimulator import DumbSimulation
from BidStrategy.AbstractBidStrategy import AbstractBidStrategy
from BidStrategy.BasicStrategy import BasicStrategy

class MyCompany(TradingCompany):
    logger: logging.Logger
    headerquarters: CompanyHeadquarters
    fleet: List[VesselWithEngine]

    # Additional Internal State
    future_trades: List[Trade]

    # Extension State
    bid_strategy: AbstractBidStrategy
    opponent_simulation: AbstractOpponentSimulation
    schedule_generator: AbstractScheduleGenerator

    def __init__(self, fleet: List[VesselWithEngine], name: str):
        super().__init__(fleet, name)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Additional Internal State
        self.future_trades = []
        
        # Extension State
        self.schedule_generator = BruteScheduleGenerator(company=self)
        self.opponent_simulation = DumbSimulation(company=self)
        self.bid_strategy = BasicStrategy(company=self)

    def pre_inform(self, trades: List[Trade], time: int):
        # Internal update
        self.future_trades = trades

        # Inform extensions
        self.schedule_generator.pre_inform(trades, time)
        self.opponent_simulation.pre_inform(trades, time)
        self.bid_strategy.pre_inform(trades, time)

    def inform(self, trades: List[Trade]) -> List[Bid]:
        # Internal update
        proposed_scheduling = self.propose_schedules(trades)

        # Inform extensions
        self.schedule_generator.inform(trades)
        self.opponent_simulation.inform(trades)
        self.bid_strategy.inform(trades)

        for vessel in self.fleet:

            self.logger.info(f"Clean schedules for vessel {vessel.name} is: {vessel.schedule.get_simple_schedule()}")
            self.logger.info(f"Untidy schedule for vessel {vessel.name} is: {vessel.schedule._get_node_locations()}")
            self.logger.info(f"Len Clean {len(vessel.schedule.get_simple_schedule())}")
            self.logger.info(f"Len Untidy {len(vessel.schedule._get_node_locations())}")
        
        # Return the bids
        return self.bid_strategy.get_bids(trades)
    
    def recieve(self, contracts: List[Contract], auction_ledger: AuctionLedger):
        # Internal update
        proposal = self.propose_schedules([contract.trade for contract in contracts])
        
        rejected_trades = self.apply_schedules(proposal.schedules)
        if rejected_trades and len(rejected_trades) > 0:
            self.logger.error(f"Rejected trades: {rejected_trades}")

        # Inform extensions
        self.schedule_generator.recieve(contracts, auction_ledger)
        self.opponent_simulation.recieve(contracts, auction_ledger)
        self.bid_strategy.recieve(contracts, auction_ledger)
 
    def propose_schedules(self, trades):
        return self.schedule_generator.generate(self, trades)