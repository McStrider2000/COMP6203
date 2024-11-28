from typing import Union, Tuple, Optional, Dict

from mable.cargo_bidding import TradingCompany
from mable.competition.information import CompanyHeadquarters
from mable.extensions.fuel_emissions import VesselWithEngine
from mable.shipping_market import Trade
from mable.simulation_space import Port

from CostEstimation import CostEstimator
import MyCompany

class FutureTradesHelper:
    def __init__(self, company: MyCompany):
        self.company = company
        self.cost_estimator = CostEstimator(company)

    def handle_future_trades(self, vessel : VesselWithEngine, trade : Trade) -> dict[str, int | Trade | float]:
        # Find the trade with the closest pickup to the current trades drop off
        closest_trade : Trade
        future_distance : int
        closest_trade, future_distance = self.find_closest_trade(trade.destination_port, self.company.future_trades, self.company.headquarters)

        trade_dict: dict[str, int | Trade | float]
        if closest_trade:
            # if dropoff_idx > 1:
            #     alt_start = schedule_locations[dropoff_idx - 2]
            # else:
            #     alt_start = vessel.location
            #
            # alt_future, alt_distance = self.find_closest_trade(alt_start, self.future_trades, self.company.headquarters)

            trade_dict = {
                'trade': closest_trade,
                'distance': future_distance,
                'estimated_cost': self.cost_estimator.estimate_fulfilment_cost(vessel, closest_trade),
                # 'distance_if_omit_trade': alt_distance
            }
        else:
            trade_dict = {}

        return trade_dict

    @staticmethod
    def find_closest_trade(starting_point : Union[Port, str], future_trades : list[Trade], hq : CompanyHeadquarters) -> Tuple[Optional[Trade], float]:
        if not future_trades:
            return None, float('inf')

        closest_future_trade: Trade = min(
            future_trades,
            key=lambda future_trade: hq.get_network_distance(
                starting_point, future_trade.origin_port
            )
        )

        distance: float = hq.get_network_distance(starting_point, closest_future_trade.origin_port)

        return closest_future_trade, distance