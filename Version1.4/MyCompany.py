from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.transport_operation import Bid
from mable.shipping_market import Trade, AuctionLedger
from mable.transport_operation import ScheduleProposal
from mable.shipping_market import Contract
from typing import List

import mable.extensions.cargo_distributions
import mable.event_management
import logging

from BruteScheduleGenerator import BruteScheduleGenerator
from OpponentTracker import OpponentTracker
import collections

class MyCompany(TradingCompany):
    def __init__(self, fleet: List[VesselWithEngine], name: str):
        super().__init__(fleet, name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._future_trades = None


        self.logger.info("Initializing MyCompany")
        self.logger.info(f"Fleet is the following: {fleet}")

        self.vessel_schedule_locations = {vessel: collections.deque() for vessel in fleet}

        # Fix the logging - use one of these options:
        # Option 1 - Using string formatting:
        self.logger.info(f"self.vessel_schedule_locations: {self.vessel_schedule_locations}")
        

        

        self.vessel_journey_log_update = {vessel: 0 for vessel in  self.fleet}

        self.trades_to_vessel_idxs = {}


        
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
            self.logger.info(f"[{i}] {vessel.name}: Location schedule={self.vessel_schedule_locations[vessel]}, Insertion points={vessel.schedule.get_insertion_points()}")
            self.logger.info(f"Actual location schedule {vessel.schedule._get_node_locations()}")
            if (list(self.vessel_schedule_locations[vessel])!=vessel.schedule._get_node_locations()):
                self.logger.error("Location schedule is not the same as the actual location schedule")
            
    def pre_inform(self, trades: List[Trade], time):
        self.remove_vessel_schedule_locations()
        self._future_trades = trades

    def inform(self, trades: List[Trade], *args, **kwargs):
        self.remove_vessel_schedule_locations()
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
        self.remove_vessel_schedule_locations()
        # Update the opponent tracker 
        self.opponent_tracker.receive(contracts, auction_ledger)
        
        
        # Trade is the trades that we have won in this current auction
        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.find_schedules(trades)
        rejected_trades = self.apply_schedules(scheduling_proposal.schedules)

        self.add_vessel_schedule_locations()


        # Log fleet after update and rejected trades
        self.log_fleet()
        if rejected_trades:
            self.logger.error(f"Rejected trades detected: {rejected_trades}")

    def remove_vessel_schedule_locations(self):
        for vessel in self.fleet:
            journey_log = vessel.journey_log
            for event in journey_log[self.vessel_journey_log_update[vessel]:]:
                if (type(event) == mable.extensions.cargo_distributions.TimeWindowArrivalEvent) or (type(event) == mable.event_management.CargoTransferEvent):
                    print(self.vessel_schedule_locations[vessel].popleft())
                    print(event)
                    print(type(event))
                    print("=====================================")

            self.vessel_journey_log_update[vessel] = len(journey_log)

    
    def add_vessel_schedule_locations(self):
        for trade in self.trades_to_vessel_idxs.keys():

            vessel, pickup_idx, delivery_idx = self.trades_to_vessel_idxs[trade]
            
            self.logger.info(f"Adding trade {trade} to vessel {vessel.name} at pickup_idx={pickup_idx}, delivery_idx={delivery_idx}")

            q = self.vessel_schedule_locations[vessel]
  
            q.insert(2*(pickup_idx - 1), trade.origin_port)
            q.insert(2*(pickup_idx - 1), trade.origin_port)
            

            adjusted_delivery_idx =2*( delivery_idx-1) + 2

            q.insert(adjusted_delivery_idx , trade.destination_port)
            q.insert(adjusted_delivery_idx , trade.destination_port)

        for vessel in self.fleet:
            if (list(self.vessel_schedule_locations[vessel])!=vessel.schedule._get_node_locations()):
                self.logger.error("Location schedule is not the same as the actual location schedule")
                self.logger.error(f"Location schedule: {self.vessel_schedule_locations[vessel]}")
                self.logger.error(f"Actual location schedule: {vessel.schedule._get_node_locations()}")
                print()
            

        self.trades_to_vessel_idxs = {}







    def find_schedules(self, trades: List[Trade]):
        scheduleProposal = self.propose_schedules(trades)
        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, scheduleProposal.costs)

    def propose_schedules(self, trades: List[Trade]):
       result, self.trades_to_vessel_idxs = self.brute_schedule_generator.generate(trades)
       return result