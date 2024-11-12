from mable.cargo_bidding import TradingCompany
from mable.examples import environment, fleets, shipping
from mable.transport_operation import Bid, ScheduleProposal


class MyCompany(TradingCompany):

    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self._future_trades = None

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
        self._future_trades = None
        return bids

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        trades = [one_contract.trade for one_contract in contracts]
        scheduling_proposal = self.find_schedules(trades)
        rejected_trades = self.apply_schedules(scheduling_proposal.schedules)

    def propose_schedules(self, trades):
        schedules = {}
        costs = {}
        scheduled_trades = []
        j = 0
        while j < len(self._fleet):
            current_vessel = self.fleet[j]
            current_vessel_schedule = schedules.get(current_vessel, current_vessel.schedule)
            new_schedule = current_vessel_schedule.copy()
            i = 0
            trade_options = {}
            while i < len(trades):
                new_schedule = current_vessel_schedule.copy()
                current_trade = trades[i]
                new_schedule.add_transportation(current_trade)
                if new_schedule.verify_schedule():
                    total_cost = self.estimate_cost(None, current_trade, current_vessel)
                    # TODO Find the closest future trade
                    # trade_options[current_trade] = ...
                    closest_future_trade, distance = self.calculate_closest_future_trade(current_trade, self._future_trades)
                    if closest_future_trade is not None:
                        trade_options[current_trade] = {
                        'cost': total_cost,
                        'min_distance_to_future': distance,
                        'closest_future_trade': closest_future_trade
                    }
                    
                i += 1
            if len(trade_options) > 0:
                selected_trade = min(
                    trade_options.keys(),
                    key=lambda trade: trade_options[trade]['min_distance_to_future']
                )

                new_schedule = current_vessel_schedule.copy()
                new_schedule.add_transportation(selected_trade)
                scheduled_trades.append(selected_trade)
                schedules[current_vessel] = new_schedule
                costs[selected_trade] = trade_options[selected_trade]['cost']


            j += 1
        return ScheduleProposal(schedules, scheduled_trades, costs)


    def calculate_closest_future_trade(self, trade, future_trades):
        min_distance = float('inf')
        closest_future_trade = None

        if self._future_trades is not None:
            for future_trade in future_trades:
                distance=self.headquarters.get_network_distance(trade.destination_port, future_trade.origin_port)
                if distance < min_distance:
                    min_distance = distance
                    closest_future_trade = future_trade

        return (closest_future_trade, min_distance)

    


    def estimate_cost(self, schedule, trade, vessel):
        hq = self.headquarters
        pick_up_travel_cost = self.calculate_travel_cost(hq, vessel, vessel.location, trade.origin_port)
        time_to_load = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_cost = vessel.get_loading_consumption(time_to_load)
        drop_off_travel_cost = self.calculate_travel_cost(hq, vessel, trade.origin_port, trade.destination_port)
        unloading_cost = vessel.get_unloading_consumption(time_to_load)

        return pick_up_travel_cost + loading_cost + drop_off_travel_cost + unloading_cost

    @staticmethod
    def calculate_travel_cost(hq, vessel, location_a, location_b, is_laden = False):
        distance_to_pickup = hq.get_network_distance(location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        if is_laden:
            return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        else:
            return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)
    def find_schedules(self, trades):
        schedules = {}
        scheduled_trades = []
        i = 0
        while i < len(trades):
            current_trade = trades[i]
            is_assigned = False
            j = 0
            while j < len(self._fleet) and not is_assigned:
                current_vessel = self._fleet[j]
                current_vessel_schedule = schedules.get(current_vessel, current_vessel.schedule)
                new_schedule = current_vessel_schedule.copy()
                new_schedule.add_transportation(current_trade)
                if new_schedule.verify_schedule():
                    schedules[current_vessel] = new_schedule
                    scheduled_trades.append(current_trade)
                    is_assigned = True
                j += 1
            i += 1
        return ScheduleProposal(schedules, scheduled_trades, {})


def build_specification():
    specifications_builder = environment.get_specification_builder(fixed_trades=shipping.example_trades_1())
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, MyCompany.__name__))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True)
    sim.run()


if __name__ == '__main__':
    build_specification()
