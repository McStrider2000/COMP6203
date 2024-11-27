import sys

from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal, Bid
from mable.examples import environment, fleets, companies
from mable.transportation_scheduling import Schedule
from dataclasses import dataclass

from printer import PrettyPrinter  

import random
from typing import List, Tuple

@dataclass
class GeneticScheduleResult:
    schedules: dict
    scheduled_trades: list
    costs: dict
    fitness: float
    generation: int


class MostBasicCompany(TradingCompany):

    def inform(self, trades, *args, **kwargs):
        return [Bid(amount=0, trade=trade) for trade in trades]




class MyCompany(TradingCompany):

    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self._future_trades = None
        self.population_size = 20
        self.generations = 10
        self.mutation_rate = 0.2
        self.elite_size = 2
        self.tournament_size = 3
        self.schedule_to_locations = {}

    def pre_inform(self, trades, time):
        self._future_trades = trades

    def inform(self, trades, *args, **kwargs):
        PrettyPrinter.print_fleet_schedules(self.fleet)
        proposed_scheduling = self.propose_schedules(trades)
        scheduled_trades = proposed_scheduling.scheduled_trades
        self._current_scheduling_proposal = proposed_scheduling
        trades_and_costs = [
            (x, proposed_scheduling.costs[x]) if x in proposed_scheduling.costs
            else (x, 0)
            for x in scheduled_trades]
        bids = [Bid(amount=cost, trade=one_trade) for one_trade, cost in trades_and_costs]
        # for bid in bids:
        #     print(f"{self.name} bidding {bid.amount} for trade {bid.trade}")
        self._future_trades = None
        return bids

    # reruns proposedschedules
    # contracts is the contracts that we have won in the auction
    # auction_ledger is the contracts that the other companies have won in the auction (index by company name)
    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        #Trade is the trades that we have won in this current auction
        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.find_schedules(trades)
        rejected_trades = self.apply_schedules(scheduling_proposal.schedules)
        PrettyPrinter.print_fleet_schedules(self.fleet)
        if rejected_trades:
            print("\033[91m====================ERROR====================")
            print("Rejected Trades Detected")
            print(f"Rejected trades: {rejected_trades}")
            print("=============================================\033[0m")


    # find the schedules for the trades for applying to the ships 
    def find_schedules(self, trades):
        scheduleProposal = self.propose_schedules(trades)

        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, scheduleProposal.costs)
    



    def propose_schedules(self, trades):        
        # if len(trades) <= 10:
        return self._propose_brute_schedules(trades)
        
        # return self._propose_genetic_schedules(trades)
    
    def _propose_brute_schedules(self, trades):
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

            orginal_lowest_cost_increase = float('inf')
            
            for vessel in self.fleet:
                curr_schedule = schedules.get(vessel, vessel.schedule)
                original_schedule = vessel.schedule
                vessel_schedule, cost_increase = self.find_cheapest_schedule(curr_schedule.copy(), trade, vessel)

                original_vessel_schedule, original_cost_increase = self.find_cheapest_schedule(original_schedule.copy(), trade, vessel)
                orginal_lowest_cost_increase= min(orginal_lowest_cost_increase, original_cost_increase)
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

            if lowest_cost_increase < orginal_lowest_cost_increase:
                lowest_cost_increase = orginal_lowest_cost_increase

            if cheapest_schedule is not None:
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = cheapest_schedule
                costs[trade] = lowest_cost_increase

        # Using the static PrettyPrinter class instead of instance method
        PrettyPrinter.print_schedule_analysis(trades, tradesToIdxs, cost_comparisons)
        return ScheduleProposal(schedules, scheduled_trades, costs)
        
    def find_cheapest_schedule(self, schedule, trade, vessel):
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
        distance_to_pickup = self.headquarters.get_network_distance(location_a, location_b)
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
    
    def get_ports_around_insertion_pair(self, schedule, vessel, idx_pickup, idx_dropoff):
        schedule_locations = self._get_schedule_locations(schedule)
        max_insertion = max(schedule.get_insertion_points())
        
        pickup_ports = self._get_insertion_ports(schedule_locations, vessel, idx_pickup, max_insertion)
        dropoff_ports = self._get_insertion_ports(schedule_locations, vessel, idx_dropoff, max_insertion)
        
        return pickup_ports + dropoff_ports
    
    def _get_schedule_locations(self, schedule):
        locations = schedule._get_node_locations()
        return [locations[i] for i in range(0, len(locations), 2)]

    def _get_insertion_ports(self, schedule_locations, vessel, idx, max_insertion):
        if idx == 1:
            return (vessel.location, schedule_locations[0] if schedule_locations else None)
        elif idx == max_insertion:
            return (schedule_locations[-1] if schedule_locations else vessel.location, None)
        else:
            return (schedule_locations[idx - 2], schedule_locations[idx - 1])
        

    
    def estimate_fulfilment_cost(self, vessel, trade) -> float:
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
	
    def calculate_travel_consumption(self, vessel, location_a, location_b, if_laden=False):
        distance_to_pickup = self.headquarters.get_network_distance(location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        if if_laden:
            return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        else:
            return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)


    # def _propose_genetic_schedules(self, trades):
    #     """Genetic algorithm based scheduling approach"""
    #     # Initialize population with random trade permutations
    #     population = self._initialize_population(trades)
    #     best_solution = None
    #     best_fitness = float('-inf')
        
    #     for generation in range(self.generations):
    #         # Evaluate current population
    #         evaluated_population = [
    #             (self._evaluate_chromosome(chromosome), chromosome) 
    #             for chromosome in population
    #         ]
    #         evaluated_population.sort(key=lambda x: x[0].fitness, reverse=True)
            
    #         # Update best solution if found
    #         if evaluated_population[0][0].fitness > best_fitness:
    #             best_fitness = evaluated_population[0][0].fitness
    #             best_solution = evaluated_population[0][0]
            
    #         # Create next generation
    #         new_population = []
            
    #         # Elitism - keep best solutions
    #         elite = [chromo for _, chromo in evaluated_population[:self.elite_size]]
    #         new_population.extend(elite)
            
    #         # Fill rest of population through selection and crossover
    #         while len(new_population) < self.population_size:
    #             parent1 = self._select_parent(evaluated_population)
    #             parent2 = self._select_parent(evaluated_population)
    #             child = self._crossover(parent1, parent2)
    #             child = self._mutate(child)
    #             new_population.append(child)
            
    #         population = new_population
        
    #     return ScheduleProposal(
    #         schedules=best_solution.schedules,
    #         scheduled_trades=best_solution.scheduled_trades,
    #         costs=best_solution.costs
    #     )
        
    # def _initialize_population(self, trades: List) -> List[List]:
    #     """Create initial population of random trade permutations"""
    #     population = []
    #     for _ in range(self.population_size):
    #         chromosome = trades.copy()
    #         random.shuffle(chromosome)
    #         population.append(chromosome)
    #     return population

    # def _evaluate_chromosome(self, chromosome: List) -> GeneticScheduleResult:
    #     """Evaluate a single chromosome (trade ordering) using greedy approach"""
    #     schedule_proposal = self._propose_brute_schedules(chromosome)
        
    #     completion_time = max(schedule.completion_time() 
    #                         for schedule in schedule_proposal.schedules.values()) if schedule_proposal.schedules else float('inf')
        
    #     total_cost = sum(schedule_proposal.costs.values())
    #     num_scheduled = len(schedule_proposal.scheduled_trades)
        
    #     # Fitness function combining multiple objectives
    #     fitness = (1000 * num_scheduled -    # Heavy weight on number of scheduled trades
    #               0.1 * completion_time -    # Small penalty for completion time
    #               0.01 * total_cost)         # Small penalty for cost
        
    #     return GeneticScheduleResult(
    #         schedules=schedule_proposal.schedules,
    #         scheduled_trades=schedule_proposal.scheduled_trades,
    #         costs=schedule_proposal.costs,
    #         fitness=fitness,
    #         generation=0
    #     )

    # def _select_parent(self, evaluated_population: List[Tuple]) -> List:
    #     """Tournament selection"""
    #     tournament = random.sample(evaluated_population, self.tournament_size)
    #     return max(tournament, key=lambda x: x[0].fitness)[1]

    # def _crossover(self, parent1: List, parent2: List) -> List:
    #     """Order crossover (OX) operator"""
    #     size = len(parent1)
    #     start, end = sorted(random.sample(range(size), 2))
        
    #     # Create child with empty spots
    #     child = [None] * size
        
    #     # Copy slice from parent1
    #     for i in range(start, end):
    #         child[i] = parent1[i]
        
    #     # Fill remaining positions with elements from parent2 in order
    #     remaining = [item for item in parent2 if item not in child[start:end]]
    #     j = 0
    #     for i in range(size):
    #         if child[i] is None:
    #             child[i] = remaining[j]
    #             j += 1
                
    #     return child

    # def _mutate(self, chromosome: List) -> List:
    #     """Swap mutation"""
    #     if random.random() < self.mutation_rate:
    #         idx1, idx2 = random.sample(range(len(chromosome)), 2)
    #         chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    #     return chromosome


def build_specification():
    number_of_month = 12
    trades_per_auction = 6
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources",
        trades_per_occurrence=trades_per_auction,
        num_auctions=number_of_month)
    my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(MyCompany.Data(MyCompany, my_fleet, MyCompany.__name__))
    for vessel in my_fleet:
        print("Vessel of mycompany",vessel.name)

    # basic_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    # specifications_builder.add_company(MostBasicCompany.Data(MostBasicCompany, basic_fleet, MostBasicCompany.__name__))
    arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=1.5))
    the_scheduler_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    for vessel in the_scheduler_fleet:
        vessel.name = "The Scheduler"+str(vessel.name)
        print("Vessel of the scheduler",vessel.name)
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=1.4))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True,
        global_agent_timeout=60)
    sim.run()






if __name__ == '__main__':
    build_specification()