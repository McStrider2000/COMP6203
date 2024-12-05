from typing import Optional
from mable.competition.information import CompanyHeadquarters
from mable.shipping_market import Trade
from mable.extensions.fuel_emissions import VesselWithEngine

def calc_fuel_to_travel(
    hq: CompanyHeadquarters, 
    vessel: VesselWithEngine, 
    location_a, location_b, 
    if_laden=False
) -> float:
    """
    Calculate the fuel consumption of a vessel traveling between two locations.
    Args:
        hq (CompanyHeadquarters): The company's headquarters.
        vessel (VesselWithEngine): The vessel to calculate fuel consumption for.
        location_a: The starting location.
        location_b: The destination location.
        if_laden (bool): Whether the vessel is laden or not.
    Returns:
        float: The predicted fuel consumption of the vessel.
    """
    distance_to_pickup = hq.get_network_distance(location_a, location_b)
    time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
    if if_laden:
        return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
    else:
        return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)
    
def calc_fuel_to_fulfill(
    hq: CompanyHeadquarters, 
    vessel: VesselWithEngine, 
    trade: Trade,
    in_transit_trade: Optional[Trade] = None,
) -> float:
    """
    Calculate the fuel consumption of a vessel fulfilling a trade. Can also account for another trade in transit.
    For in_transit_trade, we assume it has been picked up and is currently in transit.
    Args:
        hq (CompanyHeadquarters): The company's headquarters.
        vessel (VesselWithEngine): The vessel to fulfill the trade with.
        trade (Trade): The trade to fulfill.
        in_transit_trade (Optional[Trade]): The trade the vessel is currently in transit for.
    Returns:
        float: The predicted fuel consumption of the vessel to fulfill the trade.
    """
    total_fuel = 0

    # First append cost of potental transit trade & update vessel location
    if in_transit_trade:
        # Add fuel to drop off & unload transit trade
        drop_off_travel_fuel = calc_fuel_to_travel(
            hq, vessel, trade.origin_port, trade.destination_port, True)
        unloading_fuel = vessel.get_unloading_consumption(
            vessel.get_loading_time(in_transit_trade.cargo_type, in_transit_trade.amount)
        )
        total_fuel += drop_off_travel_fuel + unloading_fuel

        vessel_location = in_transit_trade.destination_port
    else: 
        vessel_location = vessel.location

    # Then append cost of fulfilling trade
    pick_up_travel_fuel = calc_fuel_to_travel(
        hq, vessel, vessel_location, trade.origin_port, False)
    loading_fuel = vessel.get_loading_consumption(
        vessel.get_loading_time(trade.cargo_type, trade.amount)
    )
    drop_off_travel_fuel = calc_fuel_to_travel(
        hq, vessel, trade.origin_port, trade.destination_port, True)
    unloading_fuel = vessel.get_unloading_consumption(
        vessel.get_loading_time(trade.cargo_type, trade.amount)
    )
    total_fuel += pick_up_travel_fuel + loading_fuel + drop_off_travel_fuel + unloading_fuel

    return total_fuel

def calc_time_to_travel(
    hq: CompanyHeadquarters, 
    vessel: VesselWithEngine, 
    location_a, location_b
) -> float:
    """
    Calculate the time taken for a vessel to travel between two locations.
    Args:
        hq (CompanyHeadquarters): The company's headquarters.
        vessel (VesselWithEngine): The vessel to calculate fuel consumption for.
        location_a: The starting location.
        location_b: The destination location.
    Returns:
        float: The predicted time taken for the vessel to travel.
    """
    distance = hq.get_network_distance(location_a, location_b)
    return vessel.get_travel_time(distance)

def calc_time_to_fulfill(
    hq: CompanyHeadquarters, 
    vessel: VesselWithEngine, 
    trade: Trade,
    in_transit_trade: Optional[Trade] = None
) -> float:
    """
    Calculate the time taken for a vessel to fulfill a trade. Can also account for another trade in transit.
    For in_transit_trade, we assume it has been picked up and is currently in transit.
    Args:
        hq (CompanyHeadquarters): The company's headquarters.
        vessel (VesselWithEngine): The vessel to fulfill the trade with.
        trade (Trade): The trade to fulfill.
        in_transit_trade (Optional[Trade]): The trade the vessel is currently in transit for.
    Returns:
        float: The predicted time taken for the vessel to fulfill the trade.
    """
    total_time = 0

    # First append time of potental transit trade & update vessel location
    if in_transit_trade:
        drop_off_time = calc_time_to_travel(
            hq, vessel, vessel.location, trade.destination_port)
        unloading_time = vessel.get_loading_time(
            in_transit_trade.cargo_type, in_transit_trade.amount)
        total_time += drop_off_time + unloading_time

        vessel_location = in_transit_trade.destination_port
    else:
        vessel_location = vessel.location

    # Then append time of fulfilling trade
    pick_up_time = calc_time_to_travel(
        hq, vessel, vessel_location, trade.origin_port)
    loading_time = vessel.get_loading_time(
        trade.cargo_type, trade.amount)
    drop_off_time = calc_time_to_travel(
        hq, vessel, trade.origin_port, trade.destination_port)
    unloading_time = loading_time
    total_time += pick_up_time + loading_time + drop_off_time + unloading_time

    return total_time


def find_closest_trade(
    hq: CompanyHeadquarters,
    current_location,
    trades: list[Trade],
) -> tuple[Trade, float]:
    """
    Find the trade with the closest pickup to the current location.
    Args:
        hq (CompanyHeadquarters): The company's headquarters.
        current_location: The current location of the vessel.
        trades (list[Trade]): The list of trades to search through.
    Returns:
        Trade: The closest trade to the current location.
        float: The distance to the closest trade.
    """
    if not trades or len(trades) < 1:
        return None, float('inf')

    closest_trade = min(
        trades,
        key=lambda future_trade: hq.get_network_distance(
            current_location, future_trade.origin_port
        )
    )
    distance = hq.get_network_distance(current_location, closest_trade.origin_port)
    return closest_trade, distance