import sys
from typing import Optional, Tuple

from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal, Bid, Trade
from mable.examples import environment, fleets, companies
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

        for trade in trades:
            trade_options = {}
            closest_trade, min_distance = self.find_closest_trade(trade, self.future_trades)
            print(closest_trade, min_distance)
            sys.exit(1)
            # chosen_vessel = None
            # shortest_schedule = None
            # for vessel in self.fleet:
            #     curr_schedule = schedules.get(vessel, vessel.schedule)
            #     if trade not in scheduled_trades:
            #         vessel_shortest_schedule = self.find_shortest_schedule(curr_schedule.copy(), trade)
            #         if vessel_shortest_schedule is not None:
            #             if (shortest_schedule is None) or (vessel_shortest_schedule.completion_time() < shortest_schedule.completion_time()):
            #                 shortest_schedule = vessel_shortest_schedule
            #                 chosen_vessel = vessel
            # if shortest_schedule is not None:
            #     scheduled_trades.append(trade)
            #     schedules[chosen_vessel] = shortest_schedule
            #     costs[trade] = self.estimate_cost(trade, chosen_vessel)

        return ScheduleProposal(schedules, scheduled_trades, costs)

    def find_closest_trade(self, trade : Trade, future_trades : list[Trade]) -> Tuple[Optional[Trade], float]:
        if not future_trades:
            return None, float('inf')

        closest_future_trade = min(
            future_trades,
            key=lambda future_trade: self.headquarters.get_network_distance(
                trade.destination_port, future_trade.origin_port
            )
        )

        distance = self.headquarters.get_network_distance(trade.destination_port, closest_future_trade.origin_port)

        return closest_future_trade, distance


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

    def estimate_cost(self, trade, vessel):
        pick_up_travel_cost = self.calculate_travel_cost(vessel, vessel.location, trade.origin_port)
        time_to_load = vessel.get_loading_time(trade.cargo_type, trade.amount)
        loading_cost = vessel.get_loading_consumption(time_to_load)
        drop_off_travel_cost = self.calculate_travel_cost(vessel, trade.origin_port, trade.destination_port)
        unloading_cost = vessel.get_unloading_consumption(time_to_load)

        return pick_up_travel_cost + loading_cost + drop_off_travel_cost + unloading_cost

    def calculate_travel_cost(self, vessel, location_a, location_b, is_laden = False):
        distance_to_pickup = self.headquarters.get_network_distance(location_a, location_b)
        time_to_pick_up = vessel.get_travel_time(distance_to_pickup)
        if is_laden:
            return vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        else:
            return vessel.get_ballast_consumption(time_to_pick_up, vessel.speed)


    @property
    def future_trades(self):
        return self._future_trades

def build_specification():
    number_of_month = 12
    trades_per_auction = 5
    specifications_builder = environment.get_specification_builder(
        environment_files_path="../resources",
        trades_per_occurrence=trades_per_auction,
        num_auctions=number_of_month)
    my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(MyCompany.Data(MyCompany, my_fleet, MyCompany.__name__))
    arch_enemy_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=1.5))
    the_scheduler_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=1.4))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True,
        global_agent_timeout=60)
    sim.run()


if __name__ == '__main__':
    build_specification()