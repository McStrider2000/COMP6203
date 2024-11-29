import sys
from collections import deque
from logging import Logger

from mable.extensions.world_ports import LatLongPort
from mable.shipping_market import Trade
from mable.cargo_bidding import TradingCompany
from mable.simulation_space import Location, Port, OnJourney
from mable.transport_operation import ScheduleProposal
from mable.transportation_scheduling import Schedule
from mable.extensions.fuel_emissions import VesselWithEngine
from typing import List, Any, Tuple
import logging

from numpy.f2py.auxfuncs import throw_error

from CostEstimation import CostEstimator
from FutureTrades import FutureTradesHelper
import MyCompany

class BruteScheduleGenerator:

    logger: Logger
    company: MyCompany
    cost_helper: CostEstimator
    future_trade_helper: FutureTradesHelper
    temp_vessel_schedule_locations: dict[VesselWithEngine, deque[Location]]

    def __init__(self, company: MyCompany, future_trades_helper: FutureTradesHelper):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.company = company
        self.cost_helper = CostEstimator(company)
        self.future_trade_helper = future_trades_helper
        self.temp_vessel_schedule_locations = {}


    def generate(self, trades: List[Trade]) -> tuple[ScheduleProposal,dict]:
        schedules: dict[VesselWithEngine, Schedule] = dict()
        scheduled_trades: list[Trade] = []
        costs: dict[Trade, float] = dict()
        cost_comparisons: dict[Trade, dict[str, float | int | dict]] = dict()

        self.temp_vessel_schedule_locations = {vessel: q.copy() for vessel, q in self.company.vessel_schedule_locations.items()}




        trades.sort(key=lambda x: x.earliest_drop_off)

        trades_to_idxs: dict[Trade, tuple[VesselWithEngine, None | int,  None | int]] = dict()

        for trade in trades:
            chosen_vessel : VesselWithEngine | None = None
            cheapest_schedule : Schedule | None = None
            lowest_cost_increase : float = float('inf')
            chosen_pickup_idx : int | None = None
            chosen_dropoff_idx : int | None = None
            chosen_vessel_schedule : Schedule | None = None

            # orginal_lowest_cost_increase = float('inf')

            for vessel in self.company.fleet:
                curr_schedule : Schedule | Any = schedules.get(vessel, vessel.schedule)
                current_vessel_schedule_deque : deque[Location] = self.temp_vessel_schedule_locations[vessel]
                if len(current_vessel_schedule_deque)>26:
                    continue
                # original_schedule = vessel.schedule

                vessel_schedule: Schedule | None
                cost_increase: float
                temp_chosen_pickup_idx: int | None
                temp_chosen_dropoff_idx: int | None
                outputed_vessel_schedule: Any | None
                vessel_schedule, cost_increase,temp_chosen_pickup_idx, temp_chosen_dropoff_idx, outputed_vessel_schedule  = self.find_cheapest_schedule(curr_schedule.copy(), trade, vessel, current_vessel_schedule_deque)

                # original_vessel_schedule, original_cost_increase, ignore , ignore2= self.find_cheapest_schedule(original_schedule.copy(), trade, vessel)
                # orginal_lowest_cost_increase= min(orginal_lowest_cost_increase, original_cost_increase)
                if vessel_schedule is not None:
                    estimated_cost: float = self.cost_helper.estimate_fulfilment_cost(vessel, trade)
                    
                    if cost_increase < lowest_cost_increase:
                        cheapest_schedule : Schedule = vessel_schedule
                        chosen_vessel : VesselWithEngine = vessel
                        lowest_cost_increase : float = cost_increase
                        chosen_pickup_idx : int = temp_chosen_pickup_idx
                        chosen_dropoff_idx : int = temp_chosen_dropoff_idx
                        chosen_vessel_schedule : deque[Location] = outputed_vessel_schedule
                        
                        
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
                trades_to_idxs[trade] = (chosen_vessel,chosen_pickup_idx, chosen_dropoff_idx)
                self.temp_vessel_schedule_locations[chosen_vessel] = chosen_vessel_schedule

        # Using the static PrettyPrinter class instead of instance method
        return ScheduleProposal(schedules, scheduled_trades, costs), trades_to_idxs

    def find_cheapest_schedule(self, schedule: Schedule, trade: Trade, vessel: VesselWithEngine, current_vessel_schedule: deque) -> tuple[Schedule | deque[Location] | None, float, float | None, float | None, Schedule | deque[Location] | None ]:
        insertion_points: list[int] = schedule.get_insertion_points()
        cheapest_schedule : Schedule | None = None
        cheapest_schedule_cost_increase : float = float('inf')
        chosen_pickup_idx : int | None = None
        chosen_dropoff_idx : int | None = None
        cheapest_schedule_deque : deque | None = None


        
        for i in range(len(insertion_points)):
            idx_pick_up : int = insertion_points[i]
            possible_drop_offs : list[int] = insertion_points[i:]
            for j in range(len(possible_drop_offs)):
                idx_drop_off : int = possible_drop_offs[j]
                schedule_option : Schedule = schedule.copy()
                current_vessel_schedule_option : deque = current_vessel_schedule.copy()
                try:
                    schedule_option.add_transportation(trade, idx_pick_up, idx_drop_off)
                    self.temp_add_vessel_schedule_locations(trade, current_vessel_schedule_option, idx_pick_up, idx_drop_off)
                    if schedule_option.verify_schedule():
                        overall_time_increase: float = schedule_option.completion_time() - schedule.completion_time()
                        time_to_trade : float = 0
                        gas_increase_travel : float = 0
                        time_to_load : float = vessel.get_loading_time(trade.cargo_type, trade.amount)
                        
                        if idx_drop_off == idx_pick_up:
                            left : Port
                            right : Port
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
                            pickup_left: Port | None
                            pickup_right: Port | None
                            dropoff_left: Port | None
                            dropoff_right: Port | None
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

                        gas_increase_loading : float = vessel.get_loading_consumption(time_to_load)
                        gas_increase_unloading : float = vessel.get_unloading_consumption(time_to_load)

                        cost_increase: float = vessel.get_cost(gas_increase_travel + gas_increase_loading + gas_increase_unloading)
                        
                        if schedule.completion_time() != 0:
                            cost_increase += max(0.0,vessel.get_cost(vessel.get_idle_consumption(overall_time_increase - time_to_trade)))
                        
                        if cost_increase < cheapest_schedule_cost_increase:
                            cheapest_schedule = schedule_option
                            cheapest_schedule_cost_increase = cost_increase
                            chosen_pickup_idx = idx_pick_up
                            chosen_dropoff_idx = idx_drop_off
                            cheapest_schedule_deque = current_vessel_schedule_option

                            
                except Exception as e:
                    print(f"Warning: Error processing schedule option: {e}")
                    raise e


        return cheapest_schedule, cheapest_schedule_cost_increase, chosen_pickup_idx, chosen_dropoff_idx, cheapest_schedule_deque

    def calc_time_to_travel(self, vessel : VesselWithEngine, location_a : Port, location_b : Port) -> float:
        distance_to_pickup : float  = self.company.headquarters.get_network_distance(
            location_a, location_b)
        time_to_pick_up: float = vessel.get_travel_time(distance_to_pickup)
        return time_to_pick_up

    def get_ports_around_insertion(self, schedule : Schedule, vessel : VesselWithEngine, idx_both : int, current_vessel_schedule : deque) -> Tuple[Location | OnJourney | Port, Location | OnJourney | Port | None]:
        schedule_locations : list[Port] = self._get_schedule_locations(current_vessel_schedule)
        max_insertion : int = max(schedule.get_insertion_points())
        
        if idx_both == 1:
            return vessel.location, schedule_locations[0] if schedule_locations else None
        elif idx_both == max_insertion:
            return schedule_locations[-1] if schedule_locations else vessel.location, None
        else:
            left_idx: int = idx_both - 2
            right_idx : int = idx_both - 1
            return schedule_locations[left_idx], schedule_locations[right_idx]

    def get_ports_around_insertion_pair(self, schedule: Schedule, vessel: VesselWithEngine, idx_pickup: int, idx_dropoff: int, current_vessel_schedule: deque[Location]) -> object:
        schedule_locations : list[Port] = self._get_schedule_locations(current_vessel_schedule)
        max_insertion : int = max(schedule.get_insertion_points())
        
        pickup_ports: tuple[Location | OnJourney, Any | None] | tuple[Location | OnJourney | Any, None] | tuple[Any, Any] = self._get_insertion_ports(schedule_locations, vessel, idx_pickup, max_insertion)
        dropoff_ports: tuple[Location | OnJourney, Any | None] | tuple[Location | OnJourney | Any, None] | tuple[Any, Any] = self._get_insertion_ports(schedule_locations, vessel, idx_dropoff, max_insertion)
        
        return pickup_ports + dropoff_ports

    @staticmethod
    def _get_schedule_locations(current_vessel_schedule_deque : deque) -> list[Port]:
        locations : deque = current_vessel_schedule_deque
        return [locations[i] for i in range(0, len(locations), 2)]

    @staticmethod
    def _get_insertion_ports(schedule_locations: list[Port], vessel: VesselWithEngine, idx: int, max_insertion: int) -> Tuple[Location | OnJourney, Port | None]:
        if idx == 1:
            return vessel.location, schedule_locations[0] if schedule_locations else None
        elif idx == max_insertion:
            return schedule_locations[-1] if schedule_locations else vessel.location, None
        else:
            return schedule_locations[idx - 2], schedule_locations[idx - 1]
        

    @staticmethod
    def temp_add_vessel_schedule_locations(trade: Trade, current_vessel_schedule_option : deque, pickup_idx: int,
                                           delivery_idx: int):
        # self.logger.info(f"Adding trade {trade} to vessel {vessel.name} at pickup_idx={pickup_idx}, delivery_idx={delivery_idx}")

        q = current_vessel_schedule_option

        q.insert(2*(pickup_idx - 1), trade.origin_port)
        q.insert(2*(pickup_idx - 1), trade.origin_port)
        

        adjusted_delivery_idx =2*( delivery_idx-1) + 2

        q.insert(adjusted_delivery_idx , trade.destination_port)
        q.insert(adjusted_delivery_idx , trade.destination_port)