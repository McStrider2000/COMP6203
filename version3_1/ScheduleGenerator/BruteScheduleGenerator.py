
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

import MyCompany
from AbstractScheduleGenerator import AbstractScheduleGenerator

class BruteScheduleGenerator(AbstractScheduleGenerator):
    
    future_trades: List[Trade]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_vessel_schedule_locations = {}

    def pre_inform(self, trades: List[Trade], time):
        self.future_trades = trades

    def generate(company, trades) -> ScheduleProposal:

        pass 


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