import sys

from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal
from mable.examples import environment, fleets
import itertools

class MyCompany(TradingCompany):
    def propose_schedules(self, trades):
        schedules = {}
        costs = {}
        scheduled_trades = []

        for vessel in self.fleet:
            curr_schedule = vessel.schedule
            for trade in trades:
                if trade not in scheduled_trades:
                    new_schedule = curr_schedule.copy()
                    insertion_points = new_schedule.get_insertion_points()
                    shortest_schedule = None
                    for i in range(len(insertion_points)):
                        idx_pick_up = insertion_points[i]
                        possible_drop_offs = insertion_points[i:]
                        for j in range(len(possible_drop_offs)):
                            idx_drop_off = possible_drop_offs[j]
                            schedule_option = new_schedule.copy()
                            schedule_option.add_transportation(trade, idx_pick_up, idx_drop_off)
                            if shortest_schedule is None or schedule_option.completion_time() < shortest_schedule.completion_time():
                                if schedule_option.verify_schedule():
                                    shortest_schedule = schedule_option
                    if shortest_schedule is not None:
                        curr_schedule = new_schedule
                        scheduled_trades.append(trade)
            schedules[vessel] = curr_schedule
        return ScheduleProposal(schedules, scheduled_trades)

if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources")
    fleet = fleets.example_fleet_2()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, "Shipping Corp Ltd."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()