from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.transport_operation import Bid
from mable.shipping_market import Trade, AuctionLedger
from mable.transport_operation import ScheduleProposal
from mable.shipping_market import Contract
from typing import List

import logging

from BruteScheduleGenerator import BruteScheduleGenerator
from OpponentTracker import OpponentTracker

class MyCompany(TradingCompany):
    def __init__(self, fleet: List[VesselWithEngine], name: str):
        super().__init__(fleet, name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._future_trades = None
        
		# New encapsulated class for stuff 
        self.brute_schedule_generator = BruteScheduleGenerator(
            company=self
        )
        self.opponent_tracker = OpponentTracker(
            company=self
        )
        
    def log_fleet(self, fleet: List[VesselWithEngine]=None):
        fleet = fleet if fleet is not None else self.fleet
        self.logger.info("Logging fleet, result:")
        for i in range(len(fleet)):
            vessel = fleet[i]
            self.logger.info(f"[{i}] {vessel.name}: Location schedule={vessel.schedule._get_node_locations()}, Insertion points={vessel.schedule.get_insertion_points()}")
            
    def pre_inform(self, trades: List[Trade], time):
        self._future_trades = trades

    def inform(self, trades: List[Trade], *args, **kwargs):
        self.log_fleet()

        # Propose a schedule and generate trades and bids based on the schedule
        proposed_scheduling = self.propose_schedules(trades)
        trades_and_costs = [
            (x, proposed_scheduling.costs[x]) if x in proposed_scheduling.costs
            else (x, 0)
            for x in proposed_scheduling.scheduled_trades]
        bids = [Bid(amount=cost, trade=one_trade) for one_trade, cost in trades_and_costs]

        # Update the current scheduling proposal and clear the future trades
        self._current_scheduling_proposal = proposed_scheduling
        self._future_trades = None

        return bids
    
    def receive(self, contracts: List[Contract], auction_ledger:AuctionLedger=None, *args, **kwargs):
        # Update the opponent tracker 
        self.opponent_tracker.receive(contracts, auction_ledger)
        
        # Trade is the trades that we have won in this current auction
        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.find_schedules(trades)
        rejected_trades = self.apply_schedules(scheduling_proposal.schedules)

        # Log fleet after update and rejected trades
        self.log_fleet()
        if rejected_trades:
            self.logger.error(f"Rejected trades detected: {rejected_trades}")

    def find_schedules(self, trades: List[Trade]):
        scheduleProposal = self.propose_schedules(trades)
        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, scheduleProposal.costs)

    def propose_schedules(self, trades: List[Trade]):
       return self.brute_schedule_generator.generate(trades)