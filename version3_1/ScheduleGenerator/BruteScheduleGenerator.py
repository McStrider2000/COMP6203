from collections import deque
from mable.shipping_market import Trade
from mable.simulation_space.universe import Location, Port, OnJourney
from mable.transport_operation import ScheduleProposal
from mable.transportation_scheduling import Schedule
from mable.extensions.fuel_emissions import VesselWithEngine
from typing import Deque, NamedTuple, Optional, Tuple, Union

from ScheduleGenerator.AbstractScheduleGenerator import AbstractScheduleGenerator
from Util import get_schedule_as_deque, calc_fuel_to_travel, calc_time_to_travel


class BruteScheduleGenerator(AbstractScheduleGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, company, trades) -> ScheduleProposal:
        """
        Generate a schedule proposal for the given company and trades.
        This is a brute force implementation that will asign trades in order of earliest drop off to vessels with the cheapest schedule.
        The cheapest schedule is 
        Args:
            company (Company): The company to generate the schedule for.
            trades (List[Trade]): The trades to schedule.
        Returns:
            ScheduleProposal: The proposed schedule.
        """

        # All setup
        schedules: dict[VesselWithEngine, Schedule] = dict()
        scheduled_trades: list[Trade] = []
        costs: dict[Trade, float] = dict()

        # Loop over trades in order of earliest drop off
        trades.sort(key=lambda x: x.earliest_drop_off)
        for trade in trades:
            # For each trade loop to find the best vessel and schedule result for the trade
            chosen_vessel: Optional[VesselWithEngine] = None
            best_result = self.CheapestScheduleResult(None, float('inf'), None)

            # Compare the chepeast schedule for the trade for each vessel
            for vessel in company.fleet:
                vessel: VesselWithEngine

                # Get the current schedule for the vessel
                curr_schedule = schedules.get(vessel, vessel.schedule)
                current_vessel_schedule_deque = get_schedule_as_deque(
                    curr_schedule)

                # Skip if the vessel already has too many?
                if len(current_vessel_schedule_deque) > 26:
                    continue

                # Find the cheapest schedule for the trade
                cheapest_schedule_result = self.find_cheapest_schedule(
                    curr_schedule.copy(), trade, vessel, current_vessel_schedule_deque)

                # If the cheapr schedule is better than the current best result, update the best result
                if cheapest_schedule_result.vessel_schedule is not None and (
                        cheapest_schedule_result.cost_increase < best_result.cost_increase):
                    chosen_vessel = vessel
                    best_result = cheapest_schedule_result

            # If the best result isn't the default, add the trade to the vessel schedule
            if best_result.cost_increase != float('inf'):
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = best_result.vessel_schedule
                costs[trade] = best_result.cost_increase
            else:
                print("no trade found")

        return ScheduleProposal(schedules, scheduled_trades, costs)

    
    class CheapestScheduleResult(NamedTuple):
        vessel_schedule: Optional[Union[Schedule, deque[Location]]]
        cost_increase: float
        outputed_vessel_schedule: Optional[Union[Schedule, deque[Location]]]
        
    def find_cheapest_schedule(
        self,
        schedule: Schedule,
        trade: Trade,
        vessel: VesselWithEngine,
        current_vessel_schedule: deque
    ) -> CheapestScheduleResult:
        """ 
        Find the cheapest schedule for a trade given a vessel and current vessel schedule.
        This is a brute force implementation that will try all possible insertion points for the trade.
        The cheapest schedule is the one with the lowest cost increase onto the original.
        Args:
            schedule (Schedule): The current schedule.
            trade (Trade): The trade to add to the schedule.
            vessel (VesselWithEngine): The vessel to add the trade to.
            current_vessel_schedule (deque): The current vessel schedule.
        Returns:
            CheapestScheduleResult: The cheapest schedule result.
        """
        cheapest_result = self.CheapestScheduleResult(None, float('inf'), None)

        # Loop over all possible insertion points
        for idx_pick_up, idx_drop_off in [
            (idx_pick_up, idx_drop_off) for i, idx_pick_up in enumerate(schedule.get_insertion_points())
            for idx_drop_off in schedule.get_insertion_points()[i:]
        ]:
            try:
                # Create a copy of the schedule
                schedule_option = schedule.copy()

                # Add the trade to the schedule & ignore if the schedule is invalid
                schedule_option.add_transportation(
                    trade, idx_pick_up, idx_drop_off)
                if not schedule_option.verify_schedule():
                    continue

                # Overall time increase and time to load
                overall_time_increase = schedule_option.completion_time() - \
                    schedule.completion_time()
                time_to_load = vessel.get_loading_time(
                    trade.cargo_type, trade.amount)

                # Then calculate the gas increase and time to trade, differentiating between the same and different insertion points
                if idx_drop_off == idx_pick_up:
                    gas_increase_travel, time_to_trade = self._calculate_for_same_insertion(
                        schedule, vessel, idx_pick_up, current_vessel_schedule, trade, time_to_load)
                else:
                    gas_increase_travel, time_to_trade = self._calculate_for_different_insertion(
                        schedule, vessel, idx_pick_up, idx_drop_off, current_vessel_schedule, trade, time_to_load)

                # Calculate the cost increase based on the total fuel consumption
                gas_increase_loading: float = vessel.get_loading_consumption(time_to_load)
                gas_increase_unloading: float = vessel.get_unloading_consumption(time_to_load)
                cost_increase: float = vessel.get_cost(
                    gas_increase_travel + gas_increase_loading + gas_increase_unloading)

                # Add idle consumption if the vessel is not already idle
                if schedule.completion_time() != 0:
                    cost_increase += max(0.0, vessel.get_cost(
                        vessel.get_idle_consumption(overall_time_increase - time_to_trade)))

                # Now check if the new schedule is the cheapest, if so update the cheapest schedule
                if cost_increase<0:
                    print("NEGATIVE COST INCREASE")
                    print("Cost increae ",cost_increase)
                    print("Gas Increase ",gas_increase_travel)
                    print("Time to trade ",time_to_trade)
                    print(trade)
                    print("For vessel ", vessel.name," with shceudle ",vessel.schedule.get_simple_schedule)
                if cost_increase < cheapest_result.cost_increase:
                    cheapest_result = self.CheapestScheduleResult(
                        schedule_option, cost_increase, current_vessel_schedule.copy())

            except Exception as e:
                self.logger.error(f"Error in find_cheapest_schedule: {e}")
                raise e
        
        # Return the cheapest schedule result
        return cheapest_result

    def _calculate_for_same_insertion(
        self,
        schedule: Schedule,
        vessel: VesselWithEngine,
        idx_pick_up: int,
        current_vessel_schedule: deque[Location],
        trade: Trade,
        time_to_load: float
    ) -> Tuple[float, float]:
        """ 
        Calculate the gas increase and time to trade for a new schedule with the added insertion points being at the same position.
        Returns:
            Tuple[float, float]: The gas increase and time to trade for the new schedule.
        """
        left: Port
        right: Port
        left, right = self._get_ports_around_same_insertion(
            schedule, vessel, idx_pick_up, current_vessel_schedule)

        time_to_trade = (time_to_load + time_to_load +
                         calc_time_to_travel(self.company.headquarters, vessel, left, trade.origin_port) +
                         calc_time_to_travel(self.company.headquarters, vessel, trade.origin_port,
                                                  trade.destination_port))
        gas_increase_travel = (calc_fuel_to_travel(self.company.headquarters, vessel, left, trade.origin_port, True) +
                               calc_fuel_to_travel(self.company.headquarters, vessel, trade.origin_port, trade.destination_port, True))

        if right is not None:
            time_to_trade += calc_time_to_travel(self.company.headquarters, vessel, trade.destination_port, right)
            gas_increase_travel += (calc_fuel_to_travel(self.company.headquarters, vessel, trade.destination_port, right, True) -
                                    calc_fuel_to_travel(self.company.headquarters, vessel, left, right, False))
        
        return gas_increase_travel, time_to_trade

    def _calculate_for_different_insertion(
        self,
        schedule: Schedule,
        vessel: VesselWithEngine,
        idx_pick_up: int,
        idx_drop_off: int,
        current_vessel_schedule: deque[Location],
        trade: Trade,
        time_to_load: float,
    ) -> Tuple[float, float]:
        pickup_left, pickup_right, dropoff_left, dropoff_right = self._get_ports_around_different_insertion(schedule, vessel, idx_pick_up, idx_drop_off,current_vessel_schedule)
                            
        time_to_trade = time_to_load + time_to_load  # Loading and unloading times
                            
        # Add pickup times
        time_to_trade += calc_time_to_travel(self.company.headquarters, vessel, pickup_left, trade.origin_port)
        if pickup_right is not None:
            time_to_trade += calc_time_to_travel(self.company.headquarters, vessel, trade.origin_port, pickup_right)
                                
        # Add dropoff times
        time_to_trade += calc_time_to_travel(self.company.headquarters, vessel, dropoff_left, trade.destination_port)
        if dropoff_right is not None:
            time_to_trade += calc_time_to_travel(self.company.headquarters, vessel, trade.destination_port, dropoff_right)

        # Calculate gas consumption for travel with None checks
        gas_increase_travel = (
            # New travel segments
            calc_fuel_to_travel(self.company.headquarters, vessel, pickup_left, trade.origin_port, False)  # To pickup
        )
                            
        if pickup_right is not None:
            gas_increase_travel += calc_fuel_to_travel(self.company.headquarters, vessel, trade.origin_port, pickup_right, True)
            gas_increase_travel -= calc_fuel_to_travel(self.company.headquarters, vessel, pickup_left, pickup_right, False)
                                
        if dropoff_right is not None:
            gas_increase_travel += (
                calc_fuel_to_travel(self.company.headquarters, vessel, dropoff_left, trade.destination_port, True) +
                calc_fuel_to_travel(self.company.headquarters, vessel, trade.destination_port, dropoff_right, False) -
                calc_fuel_to_travel(self.company.headquarters, vessel, dropoff_left, dropoff_right, False)
            )
        else:
            gas_increase_travel += calc_fuel_to_travel(self.company.headquarters, vessel, dropoff_left, trade.destination_port, True)

        return gas_increase_travel, time_to_trade

    def _get_ports_around_same_insertion(
        self, 
        schedule: Schedule, 
        vessel: VesselWithEngine, 
        idx_both: int, 
        current_vessel_schedule: deque
    ) -> Tuple[
        Union[Location, OnJourney, Port], 
        Optional[Union[Location, OnJourney, Port]]
    ]:
        """
        Get the ports around the insertion points (where they are equal).
        """
        schedule_locations: list[Port] = self._unload_schedule_deque(
            current_vessel_schedule)
        max_insertion = max(schedule.get_insertion_points())

        if idx_both == 1:
            return vessel.location, schedule_locations[0] if schedule_locations else None
        elif idx_both == max_insertion:
            return schedule_locations[-1] if schedule_locations else vessel.location, None
        else:
            left_idx: int = idx_both - 2
            right_idx: int = idx_both - 1
            return schedule_locations[left_idx], schedule_locations[right_idx]

    def _get_ports_around_different_insertion(
        self, 
        schedule: Schedule, 
        vessel: VesselWithEngine, 
        idx_pickup: int, 
        idx_dropoff: int, 
        current_vessel_schedule: Deque[Location]
    ) -> Tuple[
        Union[Location, OnJourney], 
        Optional[Port], 
        Union[Location, OnJourney], 
        Optional[Port]
    ]:
        """
        Get the ports around the insertion points (where they are different).
        """
        schedule_locations = self._unload_schedule_deque(current_vessel_schedule)
        max_insertion = max(schedule.get_insertion_points())

        pickup_ports = self._get_insertion_ports(schedule_locations, vessel, idx_pickup, max_insertion)
        dropoff_ports = self._get_insertion_ports(schedule_locations, vessel, idx_dropoff, max_insertion)

        return pickup_ports + dropoff_ports

    @staticmethod
    def _unload_schedule_deque(current_vessel_schedule_deque: deque[Location]) -> list[Port]:
        """
        Get the schedule locations from the current vessel schedule deque.
        Args:
            current_vessel_schedule_deque (deque[Location]): The current vessel schedule deque.
        Returns:
            list[Port]: The schedule locations in order.
        """
        locations: deque = current_vessel_schedule_deque
        return [locations[i] for i in range(0, len(locations), 2)]

    @staticmethod
    def _get_insertion_ports(schedule_locations: list[Port], vessel: VesselWithEngine, idx: int, max_insertion: int
    ) -> Tuple[
        Union[Location, OnJourney], 
        Optional[Port]
    ]:
        if idx == 1:
            return vessel.location, schedule_locations[0] if schedule_locations else None
        elif idx == max_insertion:
            return schedule_locations[-1] if schedule_locations else vessel.location, None
        else:
            return schedule_locations[idx - 2], schedule_locations[idx - 1]
