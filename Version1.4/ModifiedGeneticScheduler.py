import random
import sys
from collections import deque
from typing import List, Tuple

from mable.competition.information import CompanyHeadquarters
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import Trade
from mable.cargo_bidding import TradingCompany
from mable.simulation_space.universe import Location
from mable.transport_operation import ScheduleProposal
from mable.transportation_scheduling import Schedule
from pyparsing import empty

from version3_1.Util import LOGGER


class ModifiedGeneticScheduler:

    @staticmethod
    def assign_trades_to_ships_randomly(trades: list[Trade], ships: list[VesselWithEngine]):
        """
        Randomly assigns trades to ships to ensure balanced initial distribution.
        """
        random.shuffle(trades)  # Shuffle trades to randomize their order
        ship_trade_map = {ship: [] for ship in ships}

        for i, trade in enumerate(trades):
            # Assign each trade to a ship in a round-robin style
            ship = ships[i % len(ships)]
            ship_trade_map[ship].append(trade)

        return ship_trade_map

    def generate_chromosome(self, trades : list[Trade], vessel_schedule : Schedule, headquarters : CompanyHeadquarters):
        shuffled_trades = random.sample(trades, len(trades))
        schedule_distance = 0
        for trade in shuffled_trades:
            schedule_option, distance = self.find_shortest_schedule(vessel_schedule.copy(), trade, headquarters)
            if schedule_option is not None:
                vessel_schedule = schedule_option

        print(schedule_distance)
        return vessel_schedule


    def generate_population(self, trades : list[Trade], ships : list[VesselWithEngine], headquarters : CompanyHeadquarters, population_size=10):
        population = []
        for _ in range(population_size):
            ship_trade_map = self.assign_trades_to_ships_randomly(trades, ships)

            solution = []
            unassigned_trades = []
            for ship in ships:
                if len(unassigned_trades) > 0:
                    ship_trade_map[ship].extend(unassigned_trades)

                chromosome = self.generate_chromosome(ship_trade_map[ship], ship.schedule.copy(), headquarters)
                solution.append(chromosome)

                if len(ship_trade_map[ship]) > 0:
                    unassigned_trades = ship_trade_map[ship]

            population.append(solution)

        return population

    @staticmethod
    def temp_add_vessel_schedule_locations(trade: Trade, current_vessel_schedule_option: deque, pickup_idx: int,
                                           delivery_idx: int):
        q = current_vessel_schedule_option

        adjusted_delivery_idx = delivery_idx - 1
        q.insert(adjusted_delivery_idx, trade.destination_port)

        q.insert(pickup_idx - 1, trade.origin_port)

    # how to add trade into schedule and find the shortest schedule with that trade and that schedule
    def find_shortest_schedule(self, schedule : Schedule, trade : Trade, headquarters : CompanyHeadquarters):
        # Base case
        insertion_points = schedule.get_insertion_points()
        if len(insertion_points) == 1:
            temp_sched = schedule.copy()
            temp_sched.add_transportation(trade, insertion_points[0], insertion_points[0])
            if temp_sched.verify_schedule():
                return temp_sched, schedule_distance + headquarters.get_network_distance(trade.origin_port, trade.destination_port)

        schedule_ports = self.get_schedule_as_list(schedule)
        distances = self.calculate_distances(schedule_ports, trade, headquarters)

        shortest_schedule = None
        best_distance = float('inf')
        for i, pick_up_point in enumerate(insertion_points):
            incremental_distance = self.calculate_incremental_distance(
                schedule_ports, distances, trade.origin_port, i
            )

            if incremental_distance >= best_distance:
                continue

            for j, drop_off_point in enumerate(insertion_points[i:], start=i):

                total_distance = incremental_distance + self.calculate_incremental_distance(
                    schedule_ports, distances, trade.destination_port, j, origin_insert_idx=i
                )
                if total_distance >= best_distance:
                    continue

                temp_sched = schedule.copy()
                temp_sched.add_transportation(trade, pick_up_point, drop_off_point)

                if temp_sched.verify_schedule():
                    best_distance = total_distance
                    shortest_schedule = temp_sched

        return shortest_schedule, schedule_distance + best_distance

    @staticmethod
    def calculate_distances(ports, trade, headquarters):
        ports = list(set(ports + [trade.origin_port, trade.destination_port]))
        distances = {port: {} for port in ports}

        for i in range(len(ports)):
            for j in range(i, len(ports)):
                distance = headquarters.get_network_distance(ports[i], ports[j])
                distances[ports[i]][ports[j]] = distance
                distances[ports[j]][ports[i]] = distance

        return distances

    @staticmethod
    def calculate_incremental_distance(schedule_ports, distances, new_port, insert_idx, origin_insert_idx=None):
        """
        Calculate the incremental distance added by inserting a new port at a given index.
        If `origin_insert_idx` is provided, the calculation considers the impact of a prior insertion.
        """
        incremental_distance = 0

        # If not the start, add distance from the previous port to the new port
        if insert_idx > 0:
            incremental_distance += distances[schedule_ports[insert_idx - 1]][new_port]

        # If not the end, add distance from the new port to the next port
        if insert_idx < len(schedule_ports):
            incremental_distance += distances[new_port][schedule_ports[insert_idx]]

        # If not at either end, subtract the direct link being replaced
        if 0 < insert_idx < len(schedule_ports):
            incremental_distance -= distances[schedule_ports[insert_idx - 1]][schedule_ports[insert_idx]]

        # If origin port is already inserted, adjust for overlapping links
        if origin_insert_idx is not None and origin_insert_idx < insert_idx < len(schedule_ports):
            incremental_distance -= distances[schedule_ports[origin_insert_idx]][schedule_ports[insert_idx]]

        return incremental_distance

    @staticmethod
    def get_schedule_as_deque(schedule: Schedule) -> deque[Location]:
        """
        Convert a schedule to a deque of locations.
        You may notice that locations are repeated in the deque. This is due to ships having to arrive and depart from the same location.
        Args:
            schedule (Schedule): The schedule to convert.
        Returns:
            deque[Location]: A deque containing the locations that the schedule outlines.
        """
        simple_schedule: List[Tuple[str, Trade]] = schedule.get_simple_schedule()
        locations = []
        for key, trade in simple_schedule:
            if key == 'PICK_UP':
                locations.append(trade.origin_port)
                locations.append(trade.origin_port)
            elif key == 'DROP_OFF':
                locations.append(trade.destination_port)
                locations.append(trade.destination_port)
            else:
                LOGGER.error(f"Unknown key in simple schedule: {key}")
        return deque(locations)

    @staticmethod
    def get_schedule_as_list(schedule: Schedule) -> list[Location]:
        """
        Convert a schedule to a list of locations.
        Args:
            schedule (Schedule): The schedule to convert.
        Returns:
            deque[Location]: A list containing the locations that the schedule outlines.
        """
        simple_schedule: List[Tuple[str, Trade]] = schedule.get_simple_schedule()
        locations = []
        for key, trade in simple_schedule:
            if key == 'PICK_UP':
                locations.append(trade.origin_port)
            elif key == 'DROP_OFF':
                locations.append(trade.destination_port)
            else:
                LOGGER.error(f"Unknown key in simple schedule: {key}")
        return locations