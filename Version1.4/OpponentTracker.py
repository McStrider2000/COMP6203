from mable.shipping_market import AuctionLedger
from mable.shipping_market import Contract
from mable.cargo_bidding import TradingCompany
from typing import List
from mable.shipping_market import Trade, AuctionLedger
from mable.extensions.fuel_emissions import VesselWithEngine
from typing import Tuple

class OpponentTracker:
    def __init__(self, company: TradingCompany):
        self.company = company
        # Define the target levels we want to hit
        self.target_levels = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10]
        # Track current exact profit factor for each distance range
        self.distance_ranges = [
            (1, {
                'factor': 1.4, 
                'consecutive_losses': 0,
                'consecutive_wins': 0,
                'base_increment': 0.7
            }),
            (4, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),    
            (10, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (20, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (40, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (80, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (160, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (320, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (640, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (1280, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (1540, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (1800, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (2560, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (3000, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (4000, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
            (5120, {'factor': 1.4, 'consecutive_losses': 0, 'consecutive_wins': 0, 'base_increment': 0.7}),
        ]  
        
        self.trade_distance = {}
        
    def receive(self, contracts: List[Contract], auction_ledger:AuctionLedger=None):
        self.update_profit_factors(auction_ledger)

    def update_profit_factors(self, auction_ledger: dict):
        for contract in auction_ledger.get(self.company.name, []):

            print("Self.trade_distance",self.trade_distance[contract.trade],"original profit factor",self.get_profit_factor(self.trade_distance[contract.trade]))
            self.adjust_for_auction_result(self.trade_distance.get(contract.trade,0), True)
            print("Self.trade_distance",self.trade_distance[contract.trade],"new profit factor",self.get_profit_factor(self.trade_distance[contract.trade]))
        for company in self.find_competitor_companies():
            for contract in auction_ledger.get(company.name, []):
                self.adjust_for_auction_result(self.trade_distance.get(contract.trade,0), False)
            


    def check_if_BOMB_trade(self, trade: Trade):
        if trade.latest_pickup_clean < self.company.headquarters.current_time:
            print("LOOKIE HERE")
            return False
        for vessel in self.find_closest_competitor_vessels(trade).values():
            if (self.company.headquarters.current_time+self.find_rush_time(trade, vessel)) < trade.latest_pickup_clean:
                return False
            print("Ship",vessel.name,"will arrive at",trade.origin_port.name,"at",self.company.headquarters.current_time+self.find_rush_time(trade, vessel))
            print("Current time is",self.company.headquarters.current_time,"rush time is",self.find_rush_time(trade, vessel),"and latest pickup time is",trade.latest_pickup_clean)
            
        print(self.company.headquarters.current_time)
        return True

    def find_competitor_companies(self):
        return [company for company in self.company.headquarters.get_companies() if company!= self.company]
    
    def find_closest_competitor_vessels(self, trade: Trade):
        closest_vessels = {}
        for company in self.find_competitor_companies():
            closest_vessel = self.find_closest_vessel(company, trade)
            closest_vessels[company] = closest_vessel
        return closest_vessels
    
    def find_closest_vessel(self, company: TradingCompany, trade: Trade):
        closest_vessel = None
        closest_distance = float('inf')
        for vessel in company.fleet:
            distance = self.company.headquarters.get_network_distance(vessel.location, trade.origin_port)
            if distance < closest_distance:
                closest_distance = distance
                closest_vessel = vessel
        return closest_vessel
    
    def find_rush_time(self,trade,closest_vessel):
        return closest_vessel.get_travel_time(self.company.headquarters.get_network_distance(closest_vessel.location, trade.origin_port))
    
    def find_closest_competitor_vessel(self, trade: Trade) -> Tuple[VesselWithEngine, float]:
        closest_vessel = None
        closest_distance = float('inf')
        for company in self.find_competitor_companies():
            for vessel in company.fleet:
                distance = self.company.headquarters.get_network_distance(vessel.location, trade.origin_port)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_vessel = vessel
        return closest_vessel, closest_distance
    
    def get_profit_factor_for_trade(self, trade: Trade) -> float:
        closest_vessel, closest_distance = self.find_closest_competitor_vessel(trade)
        self.trade_distance[trade] = closest_distance
        profit_factor=self.get_profit_factor(closest_distance)
        print("Profit factor for trade",trade,"with distance",closest_distance,"is",profit_factor)
        return profit_factor
    
    def calculate_dynamic_increment(self, state):
        # Increase increment based on consecutive wins
        # For example: 0.1 -> 0.15 -> 0.2 -> 0.25 etc.
        return state['base_increment'] * (1 + 0.5 * (state['consecutive_wins'] // 1))
    
    def get_profit_factor(self, distance: float) -> float:
        for max_dist, state in self.distance_ranges:
            if distance <= max_dist:
                return state['factor']
        return self.distance_ranges[-1][1]['factor']
    
    def adjust_for_auction_result(self, distance: float, won_auction: bool):
        range_index = 0
        for i, (max_dist, _) in enumerate(self.distance_ranges):
            if distance <= max_dist:
                range_index = i
                break
        else:
            range_index = len(self.distance_ranges) - 1
            
        state = self.distance_ranges[range_index][1]
        current_factor = state['factor']
        
        if won_auction:
            # Increment consecutive wins and reset losses
            state['consecutive_wins'] += 1
            state['consecutive_losses'] = 0
            
            # Calculate dynamic increment based on win streak
            dynamic_increment = self.calculate_dynamic_increment(state)
            
            # Find next target level
            next_level = None
            for level in self.target_levels:
                if level > current_factor:
                    next_level = level
                    break
                    
            if next_level:
                # Increase by dynamic increment, but don't overshoot next level
                new_factor = min(current_factor + dynamic_increment, next_level)
                state['factor'] = new_factor
                self.distance_ranges[range_index] = (
                    self.distance_ranges[range_index][0],
                    state
                )
        else:
            # Reset consecutive wins and increment losses
            state['consecutive_wins'] = 0
            state['consecutive_losses'] += 1
            
            # Only drop a level if we've lost twice in a row
            if state['consecutive_losses'] >= 9:
                # Find current level index
                current_level_index = 0
                for i, level in enumerate(self.target_levels):
                    if level > current_factor:
                        current_level_index = max(0, i - 1)
                        break
                
                # Drop by one level
                new_level_index = max(0, current_level_index - 1)
                state['factor'] = self.target_levels[new_level_index]
                # Reset consecutive losses after dropping
                state['consecutive_losses'] = 0
            
            self.distance_ranges[range_index] = (
                self.distance_ranges[range_index][0],
                state
            )