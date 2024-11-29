from mable.cargo_bidding import TradingCompany
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.transport_operation import Bid
from mable.shipping_market import Trade, AuctionLedger
from mable.transport_operation import ScheduleProposal
from mable.shipping_market import Contract
from mable.competition.information import CompanyHeadquarters
from typing import List, Tuple, Union, Optional

from mable.simulation_space import Port


import mable.extensions.cargo_distributions
import mable.event_management
import logging

# from BruteScheduleGenerator import BruteScheduleGenerator
# from FutureTrades import FutureTradesHelper
# from OpponentTracker import OpponentTracker
import collections

class FakeCompany(TradingCompany):
    def __init__(self, fleet: List[VesselWithEngine], name: str):
        super().__init__(fleet, name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._future_trades = None


        self.logger.info("Initializing FakeCompany")
        self.logger.info(f"Fleet is the following: {fleet}")

        self.vessel_schedule_locations = {vessel: collections.deque() for vessel in fleet}

        # Fix the logging - use one of these options:
        # Option 1 - Using string formatting:
        self.logger.info(f"self.vessel_schedule_locations: {self.vessel_schedule_locations}")

        self.vessel_journey_log_update = {vessel: 0 for vessel in  self.fleet}

        self.trades_to_vessel_idxs = {}

        self.future_trades_helper = FutureTradesHelper(self)
        
        # New encapsulated class for stuff
        self.brute_schedule_generator = BruteScheduleGenerator(
            company=self,
            future_trades_helper=self.future_trades_helper
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
        self.future_trades = trades

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
        self.future_trades = None

        return bids
    
    def receive(self, contracts: List[Contract], auction_ledger:AuctionLedger=None, *args, **kwargs):
        print(type(auction_ledger))
        print("loookie here")
        print(auction_ledger)
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
            difference = 0
            if len(q) %2 != 0:
                difference = -1

            q.insert(2*(delivery_idx - 1)+difference , trade.destination_port)
            q.insert(2*(delivery_idx - 1)+difference , trade.destination_port)



            q.insert(2*(pickup_idx - 1)+difference , trade.origin_port)
            q.insert(2*(pickup_idx - 1)+difference , trade.origin_port)
            print(q)
            

            # adjusted_delivery_idx =2*( delivery_idx-1) + 2



        for vessel in self.fleet:
            if (list(self.vessel_schedule_locations[vessel])!=vessel.schedule._get_node_locations()):
                self.logger.error("Location schedule is not the same as the actual location schedule")
                self.logger.error(f"Location schedule: {self.vessel_schedule_locations[vessel]}")
                self.logger.error(f"Actual location schedule: {vessel.schedule._get_node_locations()}")
                print()
            

        self.trades_to_vessel_idxs = {}
    
    def check_if_BOMB_trade(self, trade: Trade):
        return self.opponent_tracker.check_if_BOMB_trade(trade)
    
    def get_profit_factor_for_trade(self, trade: Trade) -> float:
        return self.opponent_tracker.get_profit_factor_for_trade(trade)







    def find_schedules(self, trades: List[Trade]):
        scheduleProposal = self.propose_schedules(trades)
        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, scheduleProposal.costs)

    def propose_schedules(self, trades: List[Trade]):
       result, self.trades_to_vessel_idxs = self.brute_schedule_generator.generate(trades)
       return result

    @property
    def future_trades(self):
        return self._future_trades

    @future_trades.setter
    def future_trades(self, future_trades):
        self._future_trades = future_trades



class OpponentTracker:
    def __init__(self, company: TradingCompany):
        self.company = company
        # Define the target levels we want to hit
        self.target_levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 10]
        # Track current exact profit factor for each distance range
        self.distance_ranges = [
            (1, {
                'factor': 1.4, 
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'base_increment': 0.7
            }),
            (4, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),    
            (10, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (20, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (40, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (80, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (160, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (320, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (640, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (1280, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (1540, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (1800, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (2560, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (3000, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (4000, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
            (5120, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.3}),
        ] 
        
        self.trade_distance = {}
        
    def receive(self, contracts: List[Contract], auction_ledger:AuctionLedger=None):
        self.update_profit_factors(auction_ledger)

    def update_profit_factors(self, auction_ledger: dict):
        for contract in auction_ledger.get(self.company.name, []):

            print("Self.trade_distance",self.trade_distance[contract.trade],"original profit factor",self.get_profit_factor(self.trade_distance[contract.trade]))
            self.adjust_for_auction_result(self.trade_distance.get(contract.trade,0), True)
            print("Self.trade_distance",self.trade_distance[contract.trade],"new profit factor",self.get_profit_factor(self.trade_distance[contract.trade]))
        for company in self.find_competitor_companies():
            for contract in auction_ledger.get(company.name, []):
                self.adjust_for_auction_result(self.trade_distance.get(contract.trade,0), False)
            


    def check_if_BOMB_trade(self, trade: Trade):
        if trade.latest_pickup_clean < self.company.headquarters.current_time:
            print("LOOKIE HERE")
            return False
        for vessel in self.find_closest_competitor_vessels(trade).values():
            if (self.company.headquarters.current_time+self.find_rush_time(trade, vessel)) < trade.latest_pickup_clean:
                return False
            print("Ship",vessel.name,"will arrive at",trade.origin_port.name,"at",self.company.headquarters.current_time+self.find_rush_time(trade, vessel))
            print("Current time is",self.company.headquarters.current_time,"rush time is",self.find_rush_time(trade, vessel),"and latest pickup time is",trade.latest_pickup_clean)
            
        print(self.company.headquarters.current_time)
        return True

    def find_competitor_companies(self):
        return [company for company in self.company.headquarters.get_companies() if company!= self.company]
    
    def find_closest_competitor_vessels(self, trade: Trade):
        closest_vessels = {}
        for company in self.find_competitor_companies():
            closest_vessel = self.find_closest_vessel(company, trade)
            closest_vessels[company] = closest_vessel
        return closest_vessels
    
    def find_closest_vessel(self, company: TradingCompany, trade: Trade):
        closest_vessel = None
        closest_distance = float('inf')
        for vessel in company.fleet:
            distance = self.company.headquarters.get_network_distance(vessel.location, trade.origin_port)
            if distance < closest_distance:
                closest_distance = distance
                closest_vessel = vessel
        return closest_vessel
    
    def find_rush_time(self,trade,closest_vessel):
        return closest_vessel.get_travel_time(self.company.headquarters.get_network_distance(closest_vessel.location, trade.origin_port))
    
    def find_closest_competitor_vessel(self, trade: Trade) -> Tuple[VesselWithEngine, float]:
        closest_vessel = None
        closest_distance = float('inf')
        for company in self.find_competitor_companies():
            for vessel in company.fleet:
                distance = self.company.headquarters.get_network_distance(vessel.location, trade.origin_port)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_vessel = vessel
        return closest_vessel, closest_distance
    
    def get_profit_factor_for_trade(self, trade: Trade) -> float:
        closest_vessel, closest_distance = self.find_closest_competitor_vessel(trade)
        self.trade_distance[trade] = closest_distance
        profit_factor=self.get_profit_factor(closest_distance)
        print("Profit factor for trade",trade,"with distance",closest_distance,"is",profit_factor)
        return profit_factor
    
    def calculate_dynamic_increment(self, state):
        # Increase increment based on consecutive wins
        # For example: 0.1 -> 0.15 -> 0.2 -> 0.25 etc.
        return state['base_increment'] * (1 + 0.5 * (state['consecutive_wins'] // 1))
    
    def get_profit_factor(self, distance: float) -> float:
        for max_dist, state in self.distance_ranges:
            if distance <= max_dist:
                return state['factor']
        return self.distance_ranges[-1][1]['factor']
    
    def adjust_for_auction_result(self, distance: float, won_auction: bool):
        range_index = 0
        for i, (max_dist, _) in enumerate(self.distance_ranges):
            if distance <= max_dist:
                range_index = i
                break
        else:
            range_index = len(self.distance_ranges) - 1
            
        state = self.distance_ranges[range_index][1]
        current_factor = state['factor']
        
        if won_auction:
            # Increment consecutive wins and reset losses
            state['consecutive_wins'] += 1
            state['consecutive_losses'] = 0
            
            # Calculate dynamic increment based on win streak
            dynamic_increment = self.calculate_dynamic_increment(state)
            
            # Find next target level
            next_level = None
            for level in self.target_levels:
                if level > current_factor:
                    next_level = level
                    break
                    
            if next_level:
                # Increase by dynamic increment, but don't overshoot next level
                new_factor = min(current_factor + dynamic_increment, next_level)
                state['factor'] = new_factor
                self.distance_ranges[range_index] = (
                    self.distance_ranges[range_index][0],
                    state
                )
        else:
            # Reset consecutive wins and increment losses
            state['consecutive_wins'] = 0
            state['consecutive_losses'] += 1
            
            # Only drop a level if we've lost twice in a row
            if state['consecutive_losses'] >= 9:
                # Find current level index
                current_level_index = 0
                for i, level in enumerate(self.target_levels):
                    if level > current_factor:
                        current_level_index = max(0, i - 1)
                        break
                
                # Drop by one level
                new_level_index = max(0, current_level_index - 1)
                state['factor'] = self.target_levels[new_level_index]
                # Reset consecutive losses after dropping
                state['consecutive_losses'] = 0
            
            self.distance_ranges[range_index] = (
                self.distance_ranges[range_index][0],
                state
            )

class CostEstimator:
    def __init__(self, company: TradingCompany):
        self.company = company

    def estimate_fulfilment_cost(self, vessel: VesselWithEngine, trade: Trade) -> float:
        """
        Calculate the cost of fulfilling a trade given a vessel.
        Assumes that the vessel is ballast when traveling to the origin port.
        Args:
          vessel (VesselWithEngine): The vessel to fulfill the trade with.
          trade (Trade): The trade to fulfill.
        Returns:
          Prediction: The predicted cost of fulfilling the trade. Always 100% confident.
        """
        # Calculate total fuel consumption
        time_to_load = vessel.get_loading_time(trade.cargo_type, trade.amount)
        pick_up_travel_fuel = self.calculate_travel_consumption(
            vessel, vessel.location, trade.origin_port, False)
        loading_fuel = vessel.get_loading_consumption(time_to_load)
        drop_off_travel_fuel = self.calculate_travel_consumption(
            vessel, trade.origin_port, trade.destination_port, True)
        unloading_fuel = vessel.get_unloading_consumption(time_to_load)

        # Return total cost of fuel
        return vessel.get_cost(pick_up_travel_fuel + loading_fuel + drop_off_travel_fuel + unloading_fuel)

    def calculate_travel_consumption(self, vessel: VesselWithEngine, location_a, location_b, if_laden=False):
        distance_to_pickup = self.company.headquarters.get_network_distance(location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        if if_laden:
            return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        else:
            return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)
        

from logging import Logger

from mable.shipping_market import Trade
from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal
from mable.transportation_scheduling import Schedule
from mable.extensions.fuel_emissions import VesselWithEngine
from typing import List, Dict, Any, Tuple
import logging

from numpy.f2py.auxfuncs import throw_error


class BruteScheduleGenerator:

    logger: Logger
    company: FakeCompany
    cost_helper: CostEstimator
    
    temp_vessel_schedule_locations: dict[Any, Any]

    def __init__(self, company: FakeCompany, future_trades_helper):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.company = company
        self.cost_helper = CostEstimator(company)
        self.future_trade_helper = future_trades_helper
        self.temp_vessel_schedule_locations = {}


    def generate(self, trades: List[Trade]) -> tuple[ScheduleProposal,dict]:
        schedules: dict[Any | None, Schedule] = dict()
        scheduled_trades: list[Trade] = []
        costs: dict[Trade, float] = dict()
        cost_comparisons: dict[Trade, dict[str, float | int]] = dict()

        self.temp_vessel_schedule_locations = {vessel: q.copy() for vessel, q in self.company.vessel_schedule_locations.items()}




        trades.sort(key=lambda x: x.earliest_drop_off)

        trades_to_idxs: dict[Trade, tuple[Any | None, int | None, int | None]] = dict()

        for trade in trades:
            chosen_vessel : VesselWithEngine = None
            cheapest_schedule = None
            lowest_cost_increase = float('inf')
            chosen_pickup_idx=None
            chosen_dropof_idx=None
            chosen_vessel_schedule = None

            # orginal_lowest_cost_increase = float('inf')
            
            for vessel in self.company.fleet:
                curr_schedule = schedules.get(vessel, vessel.schedule)
                current_vessel_schedule = self.temp_vessel_schedule_locations[vessel]
                if len(current_vessel_schedule)>26:
                    continue
                # original_schedule = vessel.schedule
                vessel_schedule, cost_increase,temp_chosen_pickup_idx, temp_chosen_dropof_idx, outputed_vessel_schedule  = self.find_cheapest_schedule(curr_schedule.copy(), trade, vessel, current_vessel_schedule)

                # original_vessel_schedule, original_cost_increase, ignore , ignore2= self.find_cheapest_schedule(original_schedule.copy(), trade, vessel)
                # orginal_lowest_cost_increase= min(orginal_lowest_cost_increase, original_cost_increase)
                if vessel_schedule is not None:
                    estimated_cost = self.cost_helper.estimate_fulfilment_cost(vessel, trade)
                    
                    if cost_increase < lowest_cost_increase:
                        cheapest_schedule = vessel_schedule
                        chosen_vessel = vessel
                        lowest_cost_increase = cost_increase
                        chosen_pickup_idx=temp_chosen_pickup_idx
                        chosen_dropof_idx=temp_chosen_dropof_idx
                        chosen_vessel_schedule = outputed_vessel_schedule
                        
                        
                        cost_comparisons[trade] = {
                            'detailed_cost': cost_increase,
                            'estimated_cost': estimated_cost,
                            'difference': estimated_cost - cost_increase,
                            'difference_percentage': ((estimated_cost - cost_increase) / cost_increase) * 100 if cost_increase > 0 else 0
                        }

                        cost_comparisons[trade]['future_trade'] = self.future_trade_helper.handle_future_trades(vessel, trade)
                    

            # if lowest_cost_increase < orginal_lowest_cost_increase:
            #     lowest_cost_increase = orginal_lowest_cost_increase

            if cheapest_schedule is not None:
                BOMBTRADE = self.company.check_if_BOMB_trade(trade)
                if BOMBTRADE:
                    self.logger.critical(f"Trade {trade} is a BOMB trade")
                    lowest_cost_increase = 10000000000000000000
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = cheapest_schedule
                print("Profit factor for trade",self.company.get_profit_factor_for_trade(trade))
                lowest_cost_increase *= self.company.get_profit_factor_for_trade(trade)
                costs[trade] = lowest_cost_increase
                trades_to_idxs[trade] = (chosen_vessel,chosen_pickup_idx, chosen_dropof_idx)
                self.temp_vessel_schedule_locations[chosen_vessel] = chosen_vessel_schedule

        # Using the static PrettyPrinter class instead of instance method
        return ScheduleProposal(schedules, scheduled_trades, costs), trades_to_idxs

    def find_cheapest_schedule(self, schedule: Schedule, trade: Trade, vessel: VesselWithEngine,current_vessel_schedule):
        insertion_points = schedule.get_insertion_points()
        cheapest_schedule = None
        cheapest_schedule_cost_increase = float('inf')
        chosen_pickup_idx = None
        chosen_dropoff_idx = None
        cheapest_schedule_deque = None


        
        for i in range(len(insertion_points)):
            idx_pick_up = insertion_points[i]
            possible_drop_offs = insertion_points[i:]
            for j in range(len(possible_drop_offs)):
                idx_drop_off = possible_drop_offs[j]
                schedule_option = schedule.copy()
                current_vessel_schedule_option = current_vessel_schedule.copy()
                try:
                    schedule_option.add_transportation(trade, idx_pick_up, idx_drop_off)
                    self.temp_add_vessel_schedule_locations(trade, current_vessel_schedule_option, idx_pick_up, idx_drop_off)
                    if schedule_option.verify_schedule():
                        overall_time_increase = schedule_option.completion_time() - schedule.completion_time()
                        time_to_trade = 0 
                        time_to_load = vessel.get_loading_time(trade.cargo_type, trade.amount)
                        
                        if idx_drop_off == idx_pick_up:
                            left, right = self.get_ports_around_insertion(schedule, vessel, idx_pick_up,current_vessel_schedule)
                            # Handle case where right is None (end of schedule)
                            if right is None:
                                time_to_trade = (time_to_load + time_to_load + 
                                            self.calc_time_to_travel(vessel, left, trade.origin_port) +
                                            self.calc_time_to_travel(vessel, trade.origin_port, trade.destination_port))
                                gas_increase_travel = (self.cost_helper.calculate_travel_consumption(vessel, left, trade.origin_port, False) +
                                                    self.cost_helper.calculate_travel_consumption(vessel, trade.origin_port, trade.destination_port, True))
                            else:
                                time_to_trade = (time_to_load + time_to_load +
                                            self.calc_time_to_travel(vessel, left, trade.origin_port) +
                                            self.calc_time_to_travel(vessel, trade.origin_port, trade.destination_port) +
                                            self.calc_time_to_travel(vessel, trade.destination_port, right))
                                gas_increase_travel = (self.cost_helper.calculate_travel_consumption(vessel, left, trade.origin_port, False) +
                                                    self.cost_helper.calculate_travel_consumption(vessel, trade.destination_port, right, False) +
                                                    self.cost_helper.calculate_travel_consumption(vessel, trade.origin_port, trade.destination_port, True) -
                                                    self.cost_helper.calculate_travel_consumption(vessel, left, right, False))
                        else:
                            pickup_left, pickup_right, dropoff_left, dropoff_right = self.get_ports_around_insertion_pair(schedule, vessel, idx_pick_up, idx_drop_off,current_vessel_schedule)
                            
                            # Calculate time components with None checks
                            time_to_trade = time_to_load + time_to_load  # Loading and unloading times
                            
                            # Add pickup times
                            time_to_trade += self.calc_time_to_travel(vessel, pickup_left, trade.origin_port)
                            if pickup_right is not None:
                                time_to_trade += self.calc_time_to_travel(vessel, trade.origin_port, pickup_right)
                                
                            # Add dropoff times
                            time_to_trade += self.calc_time_to_travel(vessel, dropoff_left, trade.destination_port)
                            if dropoff_right is not None:
                                time_to_trade += self.calc_time_to_travel(vessel, trade.destination_port, dropoff_right)

                            # Calculate gas consumption for travel with None checks
                            gas_increase_travel = (
                                # New travel segments
                                self.cost_helper.calculate_travel_consumption(vessel, pickup_left, trade.origin_port, False)  # To pickup
                            )
                            
                            if pickup_right is not None:
                                gas_increase_travel += self.cost_helper.calculate_travel_consumption(vessel, trade.origin_port, pickup_right, True)
                                gas_increase_travel -= self.cost_helper.calculate_travel_consumption(vessel, pickup_left, pickup_right, True)
                                
                            if dropoff_right is not None:
                                gas_increase_travel += (
                                    self.cost_helper.calculate_travel_consumption(vessel, dropoff_left, trade.destination_port, True) +
                                    self.cost_helper.calculate_travel_consumption(vessel, trade.destination_port, dropoff_right, False) -
                                    self.cost_helper.calculate_travel_consumption(vessel, dropoff_left, dropoff_right, False)
                                )
                            else:
                                gas_increase_travel += self.cost_helper.calculate_travel_consumption(vessel, dropoff_left, trade.destination_port, True)

                        gas_increase_loading = vessel.get_loading_consumption(time_to_load)
                        gas_increase_unloading = vessel.get_unloading_consumption(time_to_load)


                        
                        cost_increase = vessel.get_cost(gas_increase_travel + gas_increase_loading + gas_increase_unloading)
                        
                        if schedule.completion_time() != 0:
                            cost_increase += max(0,vessel.get_cost(vessel.get_idle_consumption(overall_time_increase - time_to_trade)))
                        
                        if cost_increase < cheapest_schedule_cost_increase:
                            cheapest_schedule = schedule_option
                            cheapest_schedule_cost_increase = cost_increase
                            chosen_pickup_idx = idx_pick_up
                            chosen_dropoff_idx = idx_drop_off
                            cheapest_schedule_deque = current_vessel_schedule_option

                            
                except Exception as e:
                    print(f"Warning: Error processing schedule option: {e}")
                    raise e


        return cheapest_schedule, cheapest_schedule_cost_increase, chosen_pickup_idx, chosen_dropoff_idx,cheapest_schedule_deque

    def calc_time_to_travel(self, vessel, location_a, location_b):
        distance_to_pickup = self.company.headquarters.get_network_distance(
            location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        return time_to_pick_up

    def get_ports_around_insertion(self, schedule, vessel, idx_both,current_vessel_schedule):
        schedule_locations = self._get_schedule_locations(current_vessel_schedule)
        max_insertion = max(schedule.get_insertion_points())
        
        if idx_both == 1:
            return vessel.location, schedule_locations[0] if schedule_locations else None
        elif idx_both == max_insertion:
            return schedule_locations[-1] if schedule_locations else vessel.location, None
        else:
            left_idx = idx_both - 2
            right_idx = idx_both - 1
            return schedule_locations[left_idx], schedule_locations[right_idx]

    def get_ports_around_insertion_pair(self, schedule: Schedule, vessel: VesselWithEngine, idx_pickup, idx_dropoff,current_vessel_schedule):
        schedule_locations = self._get_schedule_locations(current_vessel_schedule)
        max_insertion = max(schedule.get_insertion_points())
        
        pickup_ports = self._get_insertion_ports(schedule_locations, vessel, idx_pickup, max_insertion)
        dropoff_ports = self._get_insertion_ports(schedule_locations, vessel, idx_dropoff, max_insertion)
        
        return pickup_ports + dropoff_ports
    
    def _get_schedule_locations(self, current_vessel_schedule):
        locations = current_vessel_schedule
        return [locations[i] for i in range(0, len(locations), 2)]
    
    def _get_insertion_ports(self, schedule_locations, vessel: VesselWithEngine, idx, max_insertion):
        if idx == 1:
            return (vessel.location, schedule_locations[0] if schedule_locations else None)
        elif idx == max_insertion:
            return (schedule_locations[-1] if schedule_locations else vessel.location, None)
        else:
            return (schedule_locations[idx - 2], schedule_locations[idx - 1])
        

    def temp_add_vessel_schedule_locations(self,trade, current_vessel_schedule_option, pickup_idx, delivery_idx):            
        # self.logger.info(f"Adding trade {trade} to vessel {vessel.name} at pickup_idx={pickup_idx}, delivery_idx={delivery_idx}")

        q = current_vessel_schedule_option

        q.insert(2*(pickup_idx - 1), trade.origin_port)
        q.insert(2*(pickup_idx - 1), trade.origin_port)
        

        adjusted_delivery_idx =2*( delivery_idx-1) + 2

        q.insert(adjusted_delivery_idx , trade.destination_port)
        q.insert(adjusted_delivery_idx , trade.destination_port)



class FutureTradesHelper:
    def __init__(self, company: FakeCompany):
        self.company = company
        self.cost_estimator = CostEstimator(company)

    def handle_future_trades(self, vessel : VesselWithEngine, trade : Trade) -> dict[str, int | Trade | float]:
        # Find the trade with the closest pickup to the current trades drop off
        closest_trade : Trade
        future_distance : int
        closest_trade, future_distance = self.find_closest_trade(trade.destination_port, self.company.future_trades, self.company.headquarters)

        trade_dict: dict[str, int | Trade | float]
        if closest_trade:
            # if dropoff_idx > 1:
            #     alt_start = schedule_locations[dropoff_idx - 2]
            # else:
            #     alt_start = vessel.location
            #
            # alt_future, alt_distance = self.find_closest_trade(alt_start, self.future_trades, self.company.headquarters)

            trade_dict = {
                'trade': closest_trade,
                'distance': future_distance,
                'estimated_cost': self.cost_estimator.estimate_fulfilment_cost(vessel, closest_trade),
                # 'distance_if_omit_trade': alt_distance
            }
        else:
            trade_dict = {}

        return trade_dict

    @staticmethod
    def find_closest_trade(starting_point : Union[Port, str], future_trades : list[Trade], hq : CompanyHeadquarters) -> Tuple[Optional[Trade], float]:
        if not future_trades:
            return None, float('inf')

        closest_future_trade: Trade = min(
            future_trades,
            key=lambda future_trade: hq.get_network_distance(
                starting_point, future_trade.origin_port
            )
        )

        distance: float = hq.get_network_distance(starting_point, closest_future_trade.origin_port)

        return closest_future_trade, distance