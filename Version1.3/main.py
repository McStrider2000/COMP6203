from mable.examples import environment, fleets, companies
from typing import Callable
import os
import logging
import colorlog
import subprocess

from MyCompany import MyCompany

def build_specification():
    logger = logging.getLogger('build_specification')
    number_of_month = 12
    trades_per_auction = 6
    specifications_builder = environment.get_specification_builder(
        environment_files_path="../resources",
        trades_per_occurrence=trades_per_auction,
        num_auctions=number_of_month
    )
    my_fleet = fleets.mixed_fleet(num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        MyCompany.Data(MyCompany, my_fleet, MyCompany.__name__))
    for vessel in my_fleet:
        logger.info(f"Vessel of mycompany {vessel.name}")

    arch_enemy_fleet = fleets.mixed_fleet(
        num_suezmax=1, num_aframax=1, num_vlcc=1)
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(
            companies.MyArchEnemy, arch_enemy_fleet, "Arch Enemy Ltd.",
            profit_factor=1.5))
    the_scheduler_fleet = fleets.mixed_fleet(
        num_suezmax=1, num_aframax=1, num_vlcc=1)
    for vessel in the_scheduler_fleet:
        vessel.name = "The Scheduler"+str(vessel.name)
        logger.info(f"Vessel of the scheduler {vessel.name}")
    specifications_builder.add_company(
        companies.TheScheduler.Data(
            companies.TheScheduler, the_scheduler_fleet, "The Scheduler LP",
            profit_factor=1.4))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True,
        global_agent_timeout=60)
    sim.run()

def main(build_specification: Callable[[], None], directory: str = '.'):

    # First remove all json files in the current directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                os.remove(os.path.join(root, file))

    # Then run the simulation
    build_specification()

    # Next try to find the json file
    json_path = next((os.path.join(root, file) for root, _, files in os.walk(
        directory) for file in files if file.endswith('.json')), None)
    if not json_path:
        return print("No json file found")

    # Then run the mable overview command
    command = ['mable', 'overview', json_path]
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