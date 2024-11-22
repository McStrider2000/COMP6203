import sys

from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal, Bid
from mable.examples import environment, fleets
import itertools

from pandas.core.config_init import pc_width_doc

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

    def find_schedules(self, trades):
        scheduleProposal = self.propose_schedules(trades)

        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, {})

    def propose_schedules(self, trades):
        schedules = {}
        costs = {}
        scheduled_trades = []

        for vessel in self.fleet:
            curr_schedule = vessel.schedule
            for trade in trades:
                if trade not in scheduled_trades:
                    # TODO: Improve cost estimation to run on each possible schedule route
                    # TODO: Do we choose shortest or cheapest?
                    cost = self.estimate_cost(curr_schedule, trade, vessel)
                    shortest_schedule = self.find_shortest_schedule(curr_schedule.copy(), trade)
                    if shortest_schedule is not None:
                        curr_schedule = shortest_schedule
                        scheduled_trades.append(trade)
                        costs[trade] = cost
            schedules[vessel] = curr_schedule
        return ScheduleProposal(schedules, scheduled_trades, costs)

    @staticmethod
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


if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources")
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, "Shipping Corp Ltd."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()