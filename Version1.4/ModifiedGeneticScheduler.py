import random
import statistics
import sys
from collections import deque
from typing import List, Tuple, Optional

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

    def generate_schedules(self, trades, ships, headquarters):
        population = self.generate_population(trades, ships, headquarters)
        fitness_scores = self.assess_population_fitness(population, headquarters)
        filtered_population, filtered_scores = self.create_temp_population(population, fitness_scores)
        self.select_edges(filtered_population, filtered_scores, headquarters)

    def generate_population(self, trades : list[Trade], ships : list[VesselWithEngine], headquarters : CompanyHeadquarters, population_size=20):
        population = []
        for _ in range(population_size):
            new_schedules, assigned_trades, unassigned_trades = self.generate_chromosome(trades, ships, headquarters)
            population.append({
                "new_schedules": new_schedules,
                "assigned_trades": assigned_trades,
                "unassigned_trades": unassigned_trades
            })

        return population

    @staticmethod
    def generate_chromosome(trades, ships, hq):
        shuffled_trades = trades
        random.shuffle(trades)
        assigned_trades = []
        unassigned_trades = []
        new_schedules = {ship: ship.schedule.copy() for ship in ships}
        for trade in shuffled_trades:
            assigned = False

            random.shuffle(ships)
            for ship in ships:
                if assigned:
                    break

                insertion_points = list(new_schedules[ship].get_insertion_points())

                if len(insertion_points) == 1:
                    temp_schedule = new_schedules[ship].copy()
                    temp_schedule.add_transportation(trade, insertion_points[0], insertion_points[0])
                    if temp_schedule.verify_schedule():
                        new_schedules[ship] = temp_schedule
                        assigned_trades.append(trade)
                        assigned = True
                        break
                else:
                    random.shuffle(insertion_points)
                    for pickup_option in insertion_points:
                        possible_dropoffs = [dropoff for dropoff in insertion_points if dropoff >= pickup_option]
                        random.shuffle(possible_dropoffs)

                        for dropoff_option in possible_dropoffs:
                            if dropoff_option >= pickup_option:
                                temp_schedule = new_schedules[ship].copy()
                                temp_schedule.add_transportation(trade, pickup_option, dropoff_option)
                                if temp_schedule.verify_schedule():
                                    new_schedules[ship] = temp_schedule
                                    assigned_trades.append(trade)
                                    assigned = True
                                    break

                        if assigned:
                            break

            if not assigned:
                unassigned_trades.append(trade)

        return new_schedules, assigned_trades, unassigned_trades

    def assess_population_fitness(self, population, headquarters, penalty = 250):
        fitness_scores = []
        for solution in population:
            travel_only_time = 0.0
            total_time = 0.0
            for ship, schedule in solution["new_schedules"].items():
                travel_only_time += self.calculate_schedule_travel_time(ship, schedule, headquarters)
                total_time += schedule.completion_time()

            missing_trades = len(solution["unassigned_trades"])
            penalty_score = penalty * missing_trades
            fitness_scores.append({
                "overall_score" : total_time + penalty_score,
                "travel_score" : travel_only_time
            })

        return fitness_scores

    def calculate_schedule_travel_time(self, ship, schedule, headquarters):
        distance = 0.0
        ports = schedule.get_simple_schedule()
        for i in range(len(ports) - 1):
            port_1 = self.retrieve_port(*ports[i])
            port_2 = self.retrieve_port(*ports[i + 1])
            distance += headquarters.get_network_distance(port_1, port_2)

        return ship.get_travel_time(distance)

    @staticmethod
    def create_temp_population(population, fitness_scores):
        average_fitness = statistics.mean([score['overall_score'] for score in fitness_scores])

        temp_population = []
        temp_scores = []
        for i in range(len(population)):
            if fitness_scores[i]['overall_score'] <= average_fitness:
                temp_population.append(population[i])
                temp_scores.append(fitness_scores[i])

        return temp_population, temp_scores

    def select_edges(self, population, fitness_scores, headquarters):
        new_population = []

        overall_scores = [fs['overall_score'] for fs in fitness_scores]
        min_score_idx = overall_scores.index(min(overall_scores))
        min_chromosome = population[min_score_idx]

        for chromosome, fitness in zip(population, fitness_scores):
            schedules = chromosome["new_schedules"]
            ships = list(schedules.keys())
            random.shuffle(ships)

            r_values = []
            selected_edges = 0

            for ship in ships:
                if selected_edges >= 3:
                    break

                schedule = schedules[ship]
                ports = schedule.get_simple_schedule()

                if len(ports) < 2:  # No edges to select if fewer than 2 ports
                    continue

                edge_index = random.randrange(len(ports) - 1)
                port_1 = self.retrieve_port(*ports[edge_index])
                port_2 = self.retrieve_port(*ports[edge_index + 1])

                distance = headquarters.get_network_distance(port_1, port_2)
                edge_cost = ship.get_travel_time(distance)

                travel_cost = fitness['travel_score']
                r_values.append(edge_cost / travel_cost)

                selected_edges += 1

            R = random.uniform(0, 1)
            if len([r for r in r_values if r <= R]) >= 2:
                new_population.append(chromosome)
            else:
                new_population.append(min_chromosome)

        return new_population


    @staticmethod
    def retrieve_port(visit_type, trade):
        return trade.origin_port if visit_type == "PICK_UP" else trade.destination_port

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

    def generate_ship_allocation(self, trades: list[Trade], vessel_schedule: Schedule,
                                 headquarters: CompanyHeadquarters):
        shuffled_trades = random.sample(trades, len(trades))
        schedule_distance = 0
        for trade in shuffled_trades:
            schedule_option, distance = self.find_shortest_schedule(vessel_schedule.copy(), trade, headquarters)
            if schedule_option is not None:
                vessel_schedule = schedule_option

        print(schedule_distance)
        return vessel_schedule

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