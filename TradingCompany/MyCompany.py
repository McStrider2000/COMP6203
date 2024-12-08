from mable.cargo_bidding import TradingCompany
from mable.transport_operation import ScheduleProposal, Bid
from mable.examples import environment, fleets
from Price import BidPredictor

class MyCompany(TradingCompany):

    def __init__(self, fleet, name):
        super().__init__(fleet, name)
        self._future_trades = None
        self._bid_predictor = BidPredictor(self.headquarters)

    def pre_inform(self, trades, time):
        self._future_trades = trades
        self._bid_predictor.pre_inform(self.headquarters, time)

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

        self._bid_predictor.receive(contracts, auction_ledger)

    def find_schedules(self, trades):
        scheduleProposal = self.propose_schedules(trades)

        return ScheduleProposal(scheduleProposal.schedules, scheduleProposal.scheduled_trades, {})

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
                costs[trade] = self._bid_predictor.estimate_fulfilment_cost(chosen_vessel, trade)

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

if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources", trades_per_occurrence=2)
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, "Shipping Corp Ltd."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()