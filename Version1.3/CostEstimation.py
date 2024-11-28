from mable.cargo_bidding import TradingCompany
from mable.competition.information import CompanyHeadquarters
from mable.shipping_market import Trade
from mable.extensions.fuel_emissions import VesselWithEngine

class CostEstimator:
    def __init__(self, company : TradingCompany):
        self.company = company

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
        distance_to_pickup = self.company.headquarters.get_network_distance(location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        if if_laden:
            return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        else:
            return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)