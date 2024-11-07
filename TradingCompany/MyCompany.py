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
                # if trade not in scheduled_trades:
                copied = curr_schedule.copy()
                copied.add_transportation(trade)
                is_valid = copied.verify_schedule()
                if is_valid:
                    curr_schedule = copied
                    scheduled_trades.append(trade)
            schedules[vessel] = curr_schedule

        return ScheduleProposal(schedules, scheduled_trades)

if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources")
    fleet = fleets.example_fleet_2()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, "Shipping Corp Ltd."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()