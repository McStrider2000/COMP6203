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
