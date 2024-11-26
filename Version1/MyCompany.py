import sys

from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal, Bid
from mable.examples import environment, fleets, companies

from dataclasses import dataclass

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

    def pre_inform(self, trades, time):
        self._future_trades = trades

    def inform(self, trades, *args, **kwargs):
        proposed_scheduling = self.propose_schedules(trades)
        scheduled_trades = proposed_scheduling.scheduled_trades
        self._current_scheduling_proposal = proposed_scheduling
        trades_and_costs = [
            (x, proposed_scheduling.costs[x]) if x in proposed_scheduling.costs
            else (x, 0)
            for x in scheduled_trades]
        bids = [Bid(amount=cost, trade=one_trade) for one_trade, cost in trades_and_costs]
        for bid in bids:
            print(f"{self.name} bidding {bid.amount} for trade {bid.trade}")
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
        if rejected_trades:
            print("====================ERROR====================")
            print("Rejected Trades Detected")
            print(f"Rejected trades: {rejected_trades}")
            print("=============================================")


    # find the schedules for the trades for applying to the ships 
    def find_schedules(self, trades):
        scheduleProposal = self.propose_schedules(trades)

        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, scheduleProposal.costs)
    



    def propose_schedules(self, trades):
        print("first shedule",self.fleet[0].schedule)
        
        if len(trades) <= 10:
            return self._propose_brute_schedules(trades)
        
        return self._propose_genetic_schedules(trades)
    
    def _propose_brute_schedules(self, trades):
        schedules = {}
        scheduled_trades = []
        costs = {}

        for trade in trades:
            chosen_vessel = None
            shortest_schedule = None
            for vessel in self.fleet:
                curr_schedule = schedules.get(vessel, vessel.schedule)
                if trade not in scheduled_trades:
                    vessel_shortest_schedule = self.find_shortest_schedule(curr_schedule.copy(), trade)
                    if vessel_shortest_schedule is not None:
                        if (shortest_schedule is None) or (vessel_shortest_schedule.completion_time() < shortest_schedule.completion_time()):
                            shortest_schedule = vessel_shortest_schedule
                            chosen_vessel = vessel
            if shortest_schedule is not None:
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = shortest_schedule
                costs[trade] = self.estimate_fulfilment_cost(chosen_vessel, trade)

        return ScheduleProposal(schedules, scheduled_trades, costs)


    def _propose_genetic_schedules(self, trades):
        """Genetic algorithm based scheduling approach"""
        # Initialize population with random trade permutations
        population = self._initialize_population(trades)
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate current population
            evaluated_population = [
                (self._evaluate_chromosome(chromosome), chromosome) 
                for chromosome in population
            ]
            evaluated_population.sort(key=lambda x: x[0].fitness, reverse=True)
            
            # Update best solution if found
            if evaluated_population[0][0].fitness > best_fitness:
                best_fitness = evaluated_population[0][0].fitness
                best_solution = evaluated_population[0][0]
            
            # Create next generation
            new_population = []
            
            # Elitism - keep best solutions
            elite = [chromo for _, chromo in evaluated_population[:self.elite_size]]
            new_population.extend(elite)
            
            # Fill rest of population through selection and crossover
            while len(new_population) < self.population_size:
                parent1 = self._select_parent(evaluated_population)
                parent2 = self._select_parent(evaluated_population)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return ScheduleProposal(
            schedules=best_solution.schedules,
            scheduled_trades=best_solution.scheduled_trades,
            costs=best_solution.costs
        )
        
    def _initialize_population(self, trades: List) -> List[List]:
        """Create initial population of random trade permutations"""
        population = []
        for _ in range(self.population_size):
            chromosome = trades.copy()
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _evaluate_chromosome(self, chromosome: List) -> GeneticScheduleResult:
        """Evaluate a single chromosome (trade ordering) using greedy approach"""
        schedule_proposal = self._propose_brute_schedules(chromosome)
        
        completion_time = max(schedule.completion_time() 
                            for schedule in schedule_proposal.schedules.values()) if schedule_proposal.schedules else float('inf')
        
        total_cost = sum(schedule_proposal.costs.values())
        num_scheduled = len(schedule_proposal.scheduled_trades)
        
        # Fitness function combining multiple objectives
        fitness = (1000 * num_scheduled -    # Heavy weight on number of scheduled trades
                  0.1 * completion_time -    # Small penalty for completion time
                  0.01 * total_cost)         # Small penalty for cost
        
        return GeneticScheduleResult(
            schedules=schedule_proposal.schedules,
            scheduled_trades=schedule_proposal.scheduled_trades,
            costs=schedule_proposal.costs,
            fitness=fitness,
            generation=0
        )

    def _select_parent(self, evaluated_population: List[Tuple]) -> List:
        """Tournament selection"""
        tournament = random.sample(evaluated_population, self.tournament_size)
        return max(tournament, key=lambda x: x[0].fitness)[1]

    def _crossover(self, parent1: List, parent2: List) -> List:
        """Order crossover (OX) operator"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child with empty spots
        child = [None] * size
        
        # Copy slice from parent1
        for i in range(start, end):
            child[i] = parent1[i]
        
        # Fill remaining positions with elements from parent2 in order
        remaining = [item for item in parent2 if item not in child[start:end]]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[j]
                j += 1
                
        return child

    def _mutate(self, chromosome: List) -> List:
        """Swap mutation"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome


    @staticmethod
    #how to add trade into schedule and find the shortest schedule with that trade and that schedule
    def find_shortest_schedule(schedule, trade):
        insertion_points = schedule.get_insertion_points()
        shortest_schedule = None
        for i in range(len(insertion_points)):
            idx_pick_up = insertion_points[i]
            possible_drop_offs = insertion_points[i:]
            for j in range(len(possible_drop_offs)):
                idx_drop_off = possible_drop_offs[j]
                schedule_option = schedule.copy()
                schedule_option.add_transportation(trade, idx_pick_up, idx_drop_off)
                if shortest_schedule is None or schedule_option.completion_time() < shortest_schedule.completion_time():
                    if schedule_option.verify_schedule():
                        shortest_schedule = schedule_option
        return shortest_schedule

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


def build_specification():
    number_of_month = 4
    trades_per_auction = 2
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