from mable.cargo_bidding import TradingCompany
from mable.competition.information import CompanyHeadquarters
from mable.examples import environment, fleets, shipping, companies
from mable.extensions.fuel_emissions import VesselWithEngine
from typing import Dict, List
from mable.shipping_market import Trade, AuctionLedger, Contract
from mable.simulation_space import Port
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDRegressor


@dataclass
class BidPredictions:
    fulfilment: float
    lowest_possible: float
    personal_history: float
    global_history: float

class BidPredictor():
    """
    A class that predicts the bids for a given trade, for each company, for each vessel.
    It will log information about prior trades and the current state of the market to make these predictions.
    """

    def __init__(self, headquarters: CompanyHeadquarters,):
        self.headquarters = headquarters
        self.time_till_next_auction = 0

        # Define the global model
        self.global_model = SGDRegressor()
        self.X = []
        self.y = []

    def pre_inform(self, headquarters, time: int) -> None:
        self.headquarters = headquarters
        self.time_till_next_auction = time - self.time_till_next_auction

    def receive(self, headquarter: CompanyHeadquarters, auction_ledger: AuctionLedger = None) -> None:
        def find_closest_distance(vessels: List[VesselWithEngine], contract: Contract) -> VesselWithEngine:
            distance = float('inf')
            for vessel in headquarter.fleet:
                new_distance = self.headquarters.get_network_distance(vessel.location, contract.trade.origin_port)
                if new_distance < distance:
                    distance = new_distance
            return distance        
        
        # Get the closest distance of every trade
        trades = []
        for headquarter in self.headquarters.get_companies():
            for trade in auction_ledger[headquarter.name]:
                distance = find_closest_distance(headquarter.fleet, trade)
                payment = trade.payment
                self.X.append([distance])
                self.y.append(payment)
                self.global_model.partial_fit([self.X[-1]], [self.y[-1]])
        pass

    
    def estimate_global_history(self, vessel: VesselWithEngine, trade: Trade) -> float:
        # If there is no data, return infinity
        if len(self.X) == 0:
            return float('inf')
        distance = self.headquarters.get_network_distance(vessel.location, trade.origin_port)
        
        # If distance is outside of the training data, return infinity
        if distance < min([x[0] for x in self.X]) or distance > max([x[0] for x in self.X]):
            return float('inf')

        predicted_bid = self.global_model.predict([[distance]])[0]
        return round(float(predicted_bid), 3)
    
    def get_competitors(self) -> List[CompanyHeadquarters]:
        return [company for company in self.headquarters.get_companies() if company != self.headquarters]

    def predict_lowest_bids(self, trades: List[Trade]) -> Dict[Trade, BidPredictions]:
        """ 
        Get the lowest bid for each trade. 
        """
        all_predictions = self.predict_all_bids(trades)
        lowest_predictions = {}

        for trade in trades:
            lowest_predictions[trade] = BidPredictions(
                float('inf'), float('inf'), float('inf'), float('inf'))
            for company in self.get_competitors():
                for vessel in company.fleet:
                    predictions = all_predictions[trade][company][vessel]
                    for key in ['fulfilment', 'lowest_possible', 'personal_history', 'global_history']:
                        if getattr(predictions, key) < getattr(lowest_predictions[trade], key):
                            pred_dict = lowest_predictions[trade].__dict__
                            pred_dict[key] = getattr(predictions, key)
                            lowest_predictions[trade] = BidPredictions(**pred_dict)

        return lowest_predictions

    def predict_all_bids(self, trades: List[Trade]) -> Dict[Trade, Dict[CompanyHeadquarters, Dict[VesselWithEngine, BidPredictions]]]:
        """
        Predict all the bids for each trade, for each company, for each vessel.
        This is a helper function for get_lowest_bids, seperated for testing purposes.
        """
        predictions = {}
        for trade in trades:
            predictions[trade] = {}
            for company in self.get_competitors():
                predictions[trade][company] = {}
                for vessel in company.fleet:
                    predictions[trade][company][vessel] = self.predict_bid(vessel, trade)
        return predictions

    def predict_bid(self, vessel: VesselWithEngine, trade: Trade) -> BidPredictions:
        return BidPredictions(
            fulfilment=self.estimate_fulfilment_cost(vessel, trade),
            lowest_possible=self.estimate_lowest_possible(vessel, trade),
            personal_history=self.estimate_personal_history(vessel, trade),
            global_history=self.estimate_global_history(vessel, trade)
        )

    def estimate_fulfilment_cost(self, vessel: VesselWithEngine, trade: Trade) -> float:
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
	
    def calculate_travel_consumption(self, vessel: VesselWithEngine, location_a, location_b, if_laden=False):
        distance_to_pickup = self.headquarters.get_network_distance(location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        if if_laden:
            return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        else:
            return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)

    def estimate_lowest_possible(self, vessel: VesselWithEngine, trade: Trade) -> float:
        """ 
        Calculate the lowest possible cost for a trade given a vessel, that doesn't cause damage to the company.
        This can be summarised as the cost of fulfillment - the cost of weighting for a better trade.
        """
        fulfillment_cost = self.estimate_fulfilment_cost(vessel, trade)

        # Currently assume they'll wait till next auction - TODO implement a better waiting time prediction
        waiting_time = self.time_till_next_auction
        waiting_cost = vessel.get_cost(
            vessel.get_idle_consumption(waiting_time))
        return fulfillment_cost - waiting_cost

    def estimate_personal_history(self, vessel: VesselWithEngine, trade: Trade) -> float:
        return float('inf')
