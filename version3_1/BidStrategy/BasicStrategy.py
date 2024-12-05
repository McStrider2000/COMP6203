from typing import List
from mable.transport_operation import Bid
from mable.shipping_market import Trade
from BidStrategy.AbstractBidStrategy import AbstractBidStrategy
from ScheduleGenerator.AbstractScheduleGenerator import AbstractScheduleGenerator

class BasicStrategy(AbstractBidStrategy):
    def get_bids(self, trades: List[Trade]) -> List[Bid]:

        schedule_generator: AbstractScheduleGenerator = self.company.schedule_generator
        proposed_scheduling = schedule_generator.generate(
            self.company, trades)
        trades_and_costs = [
            (x, proposed_scheduling.costs[x]) if x in proposed_scheduling.costs
            else (x, 0)
            for x in proposed_scheduling.scheduled_trades]
        bids = [Bid(amount=cost, trade=one_trade) for one_trade, cost in trades_and_costs]
        
        return bids