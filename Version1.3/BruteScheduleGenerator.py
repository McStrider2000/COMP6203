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
        cost_comparisons = {}  # New dictionary to store both cost calculations

        trades.sort(key=lambda x: x.earliest_drop_off)

        for trade in trades:
            chosen_vessel = None
            cheapest_schedule = None
            lowest_cost_increase = float('inf')

            for vessel in self.company.fleet:
                vessel: VesselWithEngine
                curr_schedule: Schedule = schedules.get(vessel, vessel.schedule)
                vessel_schedule, cost_increase = self.find_cheapest_schedule(
                    curr_schedule.copy(), trade, vessel)

                if vessel_schedule is not None:
                    # Compare with the estimate_fulfilment_cost
                    estimated_cost = self.estimate_fulfilment_cost(
                        vessel, trade)

                    if cost_increase < lowest_cost_increase:
                        cheapest_schedule = vessel_schedule
                        chosen_vessel = vessel
                        lowest_cost_increase = cost_increase

                        # Store both costs for comparison
                        cost_comparisons[trade] = {
                            'detailed_cost': cost_increase,
                            'estimated_cost': estimated_cost,
                            'difference': estimated_cost - cost_increase,
                            'difference_percentage': ((estimated_cost - cost_increase) / cost_increase) * 100 if cost_increase > 0 else 0
                        }

                        # Get the pickup and dropoff indices from the modified schedule
                        schedule_locations = vessel_schedule._get_node_locations()
                        schedule_locations = [schedule_locations[i]
                                              for i in range(0, len(schedule_locations), 2)]

                        pickup_idx = schedule_locations.index(
                            trade.origin_port) + 1
                        dropoff_idx = schedule_locations.index(
                            trade.destination_port) + 1
                        tradesToIdxs[trade] = (pickup_idx, dropoff_idx)

            if cheapest_schedule is not None:
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = cheapest_schedule
                costs[trade] = lowest_cost_increase

        # Print the schedule and cost comparison information
        self.logger.info("\nSchedule and Cost Analysis:")
        for trade in trades:
            if trade in tradesToIdxs:
                self.logger.info(
                    f"\nTrade: {
                        trade.origin_port} -> {trade.destination_port} | "
                    f"Schedule: Start at idx {tradesToIdxs[trade][0]}, End at idx {
                        tradesToIdxs[trade][1]}"
                )

                if trade in cost_comparisons:
                    comparison = cost_comparisons[trade]
                    self.logger.info(
                        "Cost Analysis:\n"
                        f"  Detailed cost: {
                            comparison['detailed_cost']:.2f} | "
                        f" Estimated cost: {
                            comparison['estimated_cost']:.2f} | "
                        f" Difference: {comparison['difference']:.2f} | "
                        f" Percentage: {
                            comparison['difference_percentage']:.2f}%"
                    )
            else:
                self.logger.info(
                    f"\nTrade: {
                        trade.origin_port} -> {trade.destination_port} (Could not be scheduled)"
                )

        # Calculate and print aggregate statistics
        if cost_comparisons:
            self.logger.info("\nAggregate Cost Analysis:")
            self.logger.info("-" * 80)

            avg_difference = sum(
                c['difference'] for c in cost_comparisons.values()) / len(cost_comparisons)
            avg_difference_percentage = sum(
                c['difference_percentage'] for c in cost_comparisons.values()) / len(cost_comparisons)
            max_difference = max(c['difference']
                                 for c in cost_comparisons.values())
            min_difference = min(c['difference']
                                 for c in cost_comparisons.values())

            self.logger.info(
                f"Average cost difference: {avg_difference:.2f} | "
                f"Average difference percentage: {
                    avg_difference_percentage:.2f}% | "
                f"Maximum difference: {max_difference:.2f} | "
                f"Minimum difference: {min_difference:.2f}"
            )

        return ScheduleProposal(schedules, scheduled_trades, costs)

    def find_cheapest_schedule(self, schedule: Schedule, trade: Trade, vessel: VesselWithEngine):
        cheapest_schedule = None
        cheapest_schedule_cost_increase = float('inf')

        insertion_points = schedule.get_insertion_points()
        for i in range(len(insertion_points)):
            idx_pick_up = insertion_points[i]
            for idx_drop_off in insertion_points[i:]:
                try:
                    schedule_option = schedule.copy()
                    schedule_option.add_transportation(
                        trade, idx_pick_up, idx_drop_off)
                    if not schedule_option.verify_schedule():
                        continue

                    time_to_load = vessel.get_loading_time(
                        trade.cargo_type, trade.amount)

                    if idx_drop_off == idx_pick_up:
                        time_to_trade, gas_increase_travel = self._calculate_single_point_time_and_gas(
                            schedule=schedule_option, vessel=vessel, trade=trade, time_to_load=time_to_load, idx_pick_up=idx_pick_up
                        )
                    else:
                        time_to_trade, gas_increase_travel = self._calculate_multi_point_time_and_gas(
                            schedule=schedule_option, vessel=vessel, trade=trade, time_to_load=time_to_load, idx_pick_up=idx_pick_up, idx_drop_off=idx_drop_off
                        )

                    gas_increase_loading = vessel.get_loading_consumption(time_to_load)
                    gas_increase_unloading = vessel.get_unloading_consumption(time_to_load)

                    cost_increase = vessel.get_cost(
                        gas_increase_travel + gas_increase_loading + gas_increase_unloading)
                    # cost_increase += vessel.get_cost(vessel.get_idle_consumption(overall_time_increase - time_to_trade))

                    if cost_increase < cheapest_schedule_cost_increase:
                        cheapest_schedule = schedule_option
                        cheapest_schedule_cost_increase = cost_increase

                except Exception as e:
                    self.logger.warning(
                        f"Warning: Error processing schedule option: {e}")
                    continue

        return cheapest_schedule, cheapest_schedule_cost_increase

    def _calculate_single_point_time_and_gas(self, schedule: Schedule, vessel: VesselWithEngine, trade: Trade, time_to_load, idx_pick_up):
        left, right = self.get_ports_around_insertion(
            schedule, vessel, idx_pick_up)
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

        return time_to_trade, gas_increase_travel

    def _calculate_multi_point_time_and_gas(self, schedule: Schedule, vessel: VesselWithEngine, trade: Trade, time_to_load, idx_pick_up, idx_drop_off):
        # Retrieve surrounding ports for pickup and drop-off
        pickup_left, pickup_right, dropoff_left, dropoff_right = self.get_ports_around_insertion_pair(
            schedule, vessel, idx_pick_up, idx_drop_off
        )

        # Initialize time to trade with loading and unloading times
        time_to_trade = time_to_load + time_to_load

        # Add travel times for the pickup segment
        time_to_trade += self.calc_time_to_travel(vessel, pickup_left, trade.origin_port)
        if pickup_right is not None:
            time_to_trade += self.calc_time_to_travel(vessel, trade.origin_port, pickup_right)

        # Add travel times for the drop-off segment
        time_to_trade += self.calc_time_to_travel(vessel, dropoff_left, trade.destination_port)
        if dropoff_right is not None:
            time_to_trade += self.calc_time_to_travel(vessel, trade.destination_port, dropoff_right)

        # Initialize gas consumption for travel
        gas_increase_travel = self.calculate_travel_consumption(vessel, pickup_left, trade.origin_port, False)

        # Add gas consumption for the pickup segment
        if pickup_right is not None:
            gas_increase_travel += self.calculate_travel_consumption(vessel, trade.origin_port, pickup_right, True)
            gas_increase_travel -= self.calculate_travel_consumption(vessel, pickup_left, pickup_right, True)

        # Add gas consumption for the drop-off segment
        if dropoff_right is not None:
            gas_increase_travel += (
                self.calculate_travel_consumption(vessel, dropoff_left, trade.destination_port, True) +
                self.calculate_travel_consumption(vessel, trade.destination_port, dropoff_right, False) -
                self.calculate_travel_consumption(vessel, dropoff_left, dropoff_right, True)
            )
        else:
            gas_increase_travel += self.calculate_travel_consumption(vessel, dropoff_left, trade.destination_port, True)
        return time_to_trade, gas_increase_travel

    def calc_time_to_travel(self, vessel, location_a, location_b):
        distance_to_pickup = self.company.headquarters.get_network_distance(
            location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        return time_to_pick_up

    def get_ports_around_insertion(self, schedule, vessel, idx_both):
        # Get unique ports (only START nodes)
        schedule_locations = schedule._get_node_locations()
        schedule_locations = [schedule_locations[i]
                              for i in range(0, len(schedule_locations), 2)]

        # Get maximum insertion point
        max_insertion = max(schedule.get_insertion_points())

        # Special case: idx_both is 1 (beginning of schedule)
        if idx_both == 1:
            left_port = vessel.location
            right_port = schedule_locations[0] if schedule_locations else None

        # Special case: idx_both is max insertion point
        elif idx_both == max_insertion:
            left_port = schedule_locations[-1] if schedule_locations else vessel.location
            right_port = None

        # Normal case: somewhere in middle of schedule
        else:
            left_idx = idx_both - 2  # -2 because indices in schedule_locations are shifted
            right_idx = idx_both - 1
            left_port = schedule_locations[left_idx]
            right_port = schedule_locations[right_idx]

        return left_port, right_port

    def get_ports_around_insertion_pair(self, schedule: Schedule, vessel: VesselWithEngine, idx_pickup, idx_dropoff):
        # Get unique ports (only START nodes)
        schedule_locations = schedule._get_node_locations()
        schedule_locations = [schedule_locations[i]
                              for i in range(0, len(schedule_locations), 2)]

        # Get maximum insertion point
        max_insertion = max(schedule.get_insertion_points())

        # Handle pickup points
        if idx_pickup == 1:
            pickup_left = vessel.location
            pickup_right = schedule_locations[0] if schedule_locations else None
        elif idx_pickup == max_insertion:
            pickup_left = schedule_locations[-1] if schedule_locations else vessel.location
            pickup_right = None
        else:
            # -2 because indices in schedule_locations are shifted
            pickup_left = schedule_locations[idx_pickup - 2]
            pickup_right = schedule_locations[idx_pickup - 1]

        # Handle dropoff points
        if idx_dropoff == 1:
            dropoff_left = vessel.location
            dropoff_right = schedule_locations[0] if schedule_locations else None
        elif idx_dropoff == max_insertion:
            dropoff_left = schedule_locations[-1] if schedule_locations else vessel.location
            dropoff_right = None
        else:
            dropoff_left = schedule_locations[idx_dropoff - 2]
            dropoff_right = schedule_locations[idx_dropoff - 1]

        return pickup_left, pickup_right, dropoff_left, dropoff_right
