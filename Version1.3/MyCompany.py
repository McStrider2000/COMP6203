from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.transport_operation import Bid
from mable.shipping_market import Trade
from mable.transport_operation import ScheduleProposal
from typing import List
import logging

from BruteScheduleGenerator import BruteScheduleGenerator
from GeneticScheduleGenerator import GeneticScheduleGenerator

class MyCompany(TradingCompany):
    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._future_trades = None
        
		# New encapsulated class for stuff 
        self.brute_schedule_generator = BruteScheduleGenerator(
            company=self
        )
        self.genetic_schedule_generator = GeneticScheduleGenerator(
            company=self,
            brute_schedule_generator=self.brute_schedule_generator,
            population_size=20,
            generations=10,
            mutation_rate=0.2,
            elite_size=2,
            tournament_size=3
        )
        

    def log_fleet(self, fleet: List[VesselWithEngine]=None):
        fleet = fleet if fleet is not None else self.fleet
        self.logger.info("Logging fleet, result:")
        for i in range(len(fleet)):
            vessel = fleet[i]
            self.logger.info(f"[{i}] {vessel.name}: Location schedule={vessel.schedule._get_node_locations()}, Insertion points={vessel.schedule.get_insertion_points()}")
            
    def pre_inform(self, trades, time):
        self._future_trades = trades

    def inform(self, trades, *args, **kwargs):
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
    
    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
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
        # If we have less than 10 trades, we will use the brute force schedule generator
        if len(trades) <= 10:
            return self.brute_schedule_generator.generate(trades)
        return self.genetic_schedule_generator.generate(trades)