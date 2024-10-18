from mable.cargo_bidding import TradingCompany
from mable.examples import environment, fleets


class MyCompany(TradingCompany):
    pass


if __name__ == '__main__':
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources")
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, "Chris' Super Shipping Corp Plc."))
    sim = environment.generate_simulation(specifications_builder)
    sim.run()