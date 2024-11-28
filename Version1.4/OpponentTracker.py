from mable.shipping_market import AuctionLedger
from mable.shipping_market import Contract
from mable.cargo_bidding import TradingCompany
from typing import List
from mable.shipping_market import Trade, AuctionLedger
from mable.extensions.fuel_emissions import VesselWithEngine

class OpponentTracker:
    def __init__(self, company: TradingCompany):
        self.company = company
        
    def receive(self, contracts: List[Contract], auction_ledger:AuctionLedger=None):
        pass 


    def check_if_BOMB_trade(self, trade: Trade):
        if trade.latest_pickup_clean < self.company.headquarters.current_time:
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