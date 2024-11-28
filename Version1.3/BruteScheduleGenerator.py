from mable.shipping_market import Trade
from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal
from mable.transportation_scheduling import Schedule
from mable.extensions.fuel_emissions import VesselWithEngine
from typing import List
import logging

from CostEstimation import CostEstimator


class BruteScheduleGenerator(CostEstimator):

    def __init__(self, company: TradingCompany):
        super().__init__(company=company)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self, trades: List[Trade]) -> ScheduleProposal:
        schedules = {}
        scheduled_trades = []
        costs = {}
        tradesToIdxs = {}
        cost_comparisons = {}

        trades.sort(key=lambda x: x.earliest_drop_off)

        for trade in trades:
            chosen_vessel = None
            cheapest_schedule = None
            lowest_cost_increase = float('inf')

            # orginal_lowest_cost_increase = float('inf')
            
            for vessel in self.company.fleet:
                curr_schedule = schedules.get(vessel, vessel.schedule)
                # original_schedule = vessel.schedule
                vessel_schedule, cost_increase = self.find_cheapest_schedule(curr_schedule.copy(), trade, vessel)

                # original_vessel_schedule, original_cost_increase = self.find_cheapest_schedule(original_schedule.copy(), trade, vessel)
                # orginal_lowest_cost_increase= min(orginal_lowest_cost_increase, original_cost_increase)
                if vessel_schedule is not None:
                    estimated_cost = self.estimate_fulfilment_cost(vessel, trade)
                    
                    if cost_increase < lowest_cost_increase:
                        cheapest_schedule = vessel_schedule
                        chosen_vessel = vessel
                        lowest_cost_increase = cost_increase
                        
                        cost_comparisons[trade] = {
                            'detailed_cost': cost_increase,
                            'estimated_cost': estimated_cost,
                            'difference': estimated_cost - cost_increase,
                            'difference_percentage': ((estimated_cost - cost_increase) / cost_increase) * 100 if cost_increase > 0 else 0
                        }
                        
                        schedule_locations = vessel_schedule._get_node_locations()
                        schedule_locations = [schedule_locations[i] for i in range(0, len(schedule_locations), 2)]
                        
                        pickup_idx = schedule_locations.index(trade.origin_port) + 1
                        dropoff_idx = schedule_locations.index(trade.destination_port) + 1
                        tradesToIdxs[trade] = (pickup_idx, dropoff_idx)

            # if lowest_cost_increase < orginal_lowest_cost_increase:
            #     lowest_cost_increase = orginal_lowest_cost_increase

            if cheapest_schedule is not None:
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = cheapest_schedule
                costs[trade] = lowest_cost_increase

        # Using the static PrettyPrinter class instead of instance method
        return ScheduleProposal(schedules, scheduled_trades, costs)

    def find_cheapest_schedule(self, schedule: Schedule, trade: Trade, vessel: VesselWithEngine):
        insertion_points = schedule.get_insertion_points()
        cheapest_schedule = None
        cheapest_schedule_cost_increase = float('inf')
        
        for i in range(len(insertion_points)):
            idx_pick_up = insertion_points[i]
            possible_drop_offs = insertion_points[i:]
            for j in range(len(possible_drop_offs)):
                idx_drop_off = possible_drop_offs[j]
                schedule_option = schedule.copy()
                try:
                    schedule_option.add_transportation(trade, idx_pick_up, idx_drop_off)
                    if schedule_option.verify_schedule():
                        overall_time_increase = schedule_option.completion_time() - schedule.completion_time()
                        time_to_trade = 0 
                        time_to_load = vessel.get_loading_time(trade.cargo_type, trade.amount)
                        
                        if idx_drop_off == idx_pick_up:
                            left, right = self.get_ports_around_insertion(schedule, vessel, idx_pick_up)
                            # Handle case where right is None (end of schedule)
                            if right is None:
                                time_to_trade = (time_to_load + time_to_load + 
                                            self.calc_time_to_travel(vessel, left, trade.origin_port) +
                                            self.calc_time_to_travel(vessel, trade.origin_port, trade.destination_port))
                                gas_increase_travel = (self.calculate_travel_consumption(vessel, left, trade.origin_port, False) + 
                                                    self.calculate_travel_consumption(vessel, trade.origin_port, trade.destination_port, True))
                            else:
                                time_to_trade = (time_to_load + time_to_load +
                                            self.calc_time_to_travel(vessel, left, trade.origin_port) +
                                            self.calc_time_to_travel(vessel, trade.origin_port, trade.destination_port) +
                                            self.calc_time_to_travel(vessel, trade.destination_port, right))
                                gas_increase_travel = (self.calculate_travel_consumption(vessel, left, trade.origin_port, False) + 
                                                    self.calculate_travel_consumption(vessel, trade.destination_port, right, False) + 
                                                    self.calculate_travel_consumption(vessel, trade.origin_port, trade.destination_port, True) - 
                                                    self.calculate_travel_consumption(vessel, left, right, True))
                        else:
                            pickup_left, pickup_right, dropoff_left, dropoff_right = self.get_ports_around_insertion_pair(schedule, vessel, idx_pick_up, idx_drop_off)
                            
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
                                self.calculate_travel_consumption(vessel, pickup_left, trade.origin_port, False)  # To pickup
                            )
                            
                            if pickup_right is not None:
                                gas_increase_travel += self.calculate_travel_consumption(vessel, trade.origin_port, pickup_right, True)
                                gas_increase_travel -= self.calculate_travel_consumption(vessel, pickup_left, pickup_right, True)
                                
                            if dropoff_right is not None:
                                gas_increase_travel += (
                                    self.calculate_travel_consumption(vessel, dropoff_left, trade.destination_port, True) +
                                    self.calculate_travel_consumption(vessel, trade.destination_port, dropoff_right, False) -
                                    self.calculate_travel_consumption(vessel, dropoff_left, dropoff_right, True)
                                )
                            else:
                                gas_increase_travel += self.calculate_travel_consumption(vessel, dropoff_left, trade.destination_port, True)

                        gas_increase_loading = vessel.get_loading_consumption(time_to_load)
                        gas_increase_unloading = vessel.get_unloading_consumption(time_to_load)
                        
                        cost_increase = vessel.get_cost(gas_increase_travel + gas_increase_loading + gas_increase_unloading)
                        
                        if schedule.completion_time() != 0:
                            cost_increase += max(0,vessel.get_cost(vessel.get_idle_consumption(overall_time_increase - time_to_trade)))
                        
                        if cost_increase < cheapest_schedule_cost_increase:
                            cheapest_schedule = schedule_option
                            cheapest_schedule_cost_increase = cost_increase
                            
                except Exception as e:
                    print(f"Warning: Error processing schedule option: {e}")
                    continue

        return cheapest_schedule, cheapest_schedule_cost_increase

    def calc_time_to_travel(self, vessel, location_a, location_b):
        distance_to_pickup = self.company.headquarters.get_network_distance(
            location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        return time_to_pick_up

    def get_ports_around_insertion(self, schedule, vessel, idx_both):
        schedule_locations = self._get_schedule_locations(schedule)
        max_insertion = max(schedule.get_insertion_points())
        
        if idx_both == 1:
            return vessel.location, schedule_locations[0] if schedule_locations else None
        elif idx_both == max_insertion:
            return schedule_locations[-1] if schedule_locations else vessel.location, None
        else:
            left_idx = idx_both - 2
            right_idx = idx_both - 1
            return schedule_locations[left_idx], schedule_locations[right_idx]

    def get_ports_around_insertion_pair(self, schedule: Schedule, vessel: VesselWithEngine, idx_pickup, idx_dropoff):
        schedule_locations = self._get_schedule_locations(schedule)
        max_insertion = max(schedule.get_insertion_points())
        
        pickup_ports = self._get_insertion_ports(schedule_locations, vessel, idx_pickup, max_insertion)
        dropoff_ports = self._get_insertion_ports(schedule_locations, vessel, idx_dropoff, max_insertion)
        
        return pickup_ports + dropoff_ports

    def _get_schedule_locations(self, schedule: Schedule):
        locations = schedule._get_node_locations()
        return [locations[i] for i in range(0, len(locations), 2)]
    
    def _get_insertion_ports(self, schedule_locations, vessel: VesselWithEngine, idx, max_insertion):
        if idx == 1:
            return (vessel.location, schedule_locations[0] if schedule_locations else None)
        elif idx == max_insertion:
            return (schedule_locations[-1] if schedule_locations else vessel.location, None)
        else:
            return (schedule_locations[idx - 2], schedule_locations[idx - 1])