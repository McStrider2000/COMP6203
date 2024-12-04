from mable.examples import environment, fleets, companies
from mable.cargo_bidding import TradingCompany
from typing import Callable
import os
import logging
import colorlog
import subprocess

from MyCompany import MyCompany

def build_specification():
    # TODO: Implement the build_specification function
    pass

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