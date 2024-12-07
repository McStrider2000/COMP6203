from mable.examples import environment, fleets, companies
from typing import Callable
import os
import logging
import colorlog
import subprocess
import glob

from OtherCompanies.MostBasicCompany import MostBasicCompany
from MyCompany import MyCompany

LOGGER = logging.getLogger(__name__)

def build_specification(
    # Environment parameters
    environment_files_path="../resources",
    # Auction parameters
    number_of_month: int = 12,
    trades_per_auction: int = 7,
    # Number of vessels
    num_suezmax: int = 1,
    num_aframax: int = 1,
    num_vlcc: int = 1,
    # Simulation parameters
    show_detailed_auction_outcome: bool = True,
    global_agent_timeout: int = 60,
    output_directory: str = '.',
):
    # Create the environment
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources",
        trades_per_occurrence=trades_per_auction,
        num_auctions=number_of_month)

    # Create our company
    my_company = MyCompany.Data(
        current_class=MyCompany, 
        fleet=fleets.mixed_fleet(num_suezmax=num_suezmax, num_aframax=num_aframax, num_vlcc=num_vlcc), 
        name=MyCompany.__name__)
    basic_company = MostBasicCompany.Data(
        current_class=MostBasicCompany, 
        fleet=fleets.mixed_fleet(num_suezmax=num_suezmax, num_aframax=num_aframax, num_vlcc=num_vlcc), 
        name=MostBasicCompany.__name__
    )
    arch_enemy_company = companies.MyArchEnemy.Data(
        current_class=companies.MyArchEnemy,
        fleet=fleets.mixed_fleet(num_suezmax=num_suezmax, num_aframax=num_aframax, num_vlcc=num_vlcc),
        name="Arch Enemy Ltd.",
        profit_factor=2.1
    )
    the_scheduler_company = companies.TheScheduler.Data(
        current_class=companies.TheScheduler,
        fleet=fleets.mixed_fleet(num_suezmax=num_suezmax, num_aframax=num_aframax, num_vlcc=num_vlcc),
        name="The Scheduler LP",
        profit_factor=2.5
    )

    # Add the companies to the simulation
    specifications_builder.add_company(my_company)
    specifications_builder.add_company(basic_company)
    specifications_builder.add_company(arch_enemy_company)
    specifications_builder.add_company(the_scheduler_company)

    # Generate the simulation
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=show_detailed_auction_outcome,
        global_agent_timeout=global_agent_timeout
    )
    sim.run()

    

def old_build_specification(output_directory: str = '.'):
    specifications_builder = environment.get_specification_builder(environment_files_path="../resources")
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleets.example_fleet_1(), "Shipping Corp Ltd."))
    sim = environment.generate_simulation(
        specifications_builder, output_directory=output_directory)
    sim.run()


def main(build_specification: Callable[[], None], output_directory: str = '.', delete_resources=True):
    # First run build specification
    build_specification()

    # Next delete the resources that are now not needed
    files = [
        'port_cargo_weight_distribution.csv',
        'port_trade_frequency_distribution.csv',
        'ports.csv',
        'precomputed_routes.pickle',
        'routing_graph_world_mask.pkl',
        'time_transition_distribution.csv',
    ]
    if delete_resources:
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    # Next try to find the output json file
    search_pattern = os.path.join(output_directory, "metrics_competition*.json")
    files = glob.glob(search_pattern)
    if not files:
        return print("No output file found")
    old_file = files[0]
    new_file_path = os.path.join(output_directory, 'output.json')

    # Rename the file to output.json
    if os.path.exists(new_file_path):
        os.remove(new_file_path)
    os.rename(old_file, new_file_path)

    # Then run the mable overview command
    command = ['mable', 'overview', new_file_path]
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Failed to execute {' '.join(command)}")
    else:
        print(result.stdout)


def setup_logging(level=logging.INFO, log_file='logs.txt'):
    logger = logging.getLogger()
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    formatter = colorlog.ColoredFormatter(
        '%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors=log_colors
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)


if __name__ == "__main__":
    setup_logging(level=logging.INFO, log_file='logs.txt')
    main(build_specification)