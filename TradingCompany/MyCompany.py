import sys

from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal
from mable.examples import environment, fleets
import itertools

from pandas.core.config_init import pc_width_doc

class MyCompany(TradingCompany):
    def propose_schedules(self, trades):
        schedules = {}
        costs = {}
        scheduled_trades = []

        for trade in trades:
            chosen_vessel = None
            shortest_schedule = None
            for vessel in self.fleet:
                curr_schedule = schedules.get(vessel, vessel.schedule)
                if trade not in scheduled_trades:
                    vessel_shortest_schedule = self.find_shortest_schedule(curr_schedule.copy(), trade)
                    if vessel_shortest_schedule is not None:
                        if (shortest_schedule is None) or (vessel_shortest_schedule.completion_time() < shortest_schedule.completion_time()):
                            shortest_schedule = vessel_shortest_schedule
                            chosen_vessel = vessel
            if shortest_schedule is not None:
                scheduled_trades.append(trade)
                schedules[chosen_vessel] = shortest_schedule
                costs[trade] = self.estimate_cost(trade, chosen_vessel)

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


if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources", trades_per_occurrence=2)
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, "Shipping Corp Ltd."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()