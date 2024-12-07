from mable.examples import environment, fleets, companies
from mable.cargo_bidding import TradingCompany
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
import subprocess
from collections import defaultdict
from prettytable import PrettyTable

@dataclass
class TournamentConfig:
    num_months: int = 12
    trades_per_auction_range: List[int] = None
    num_rounds: int = 3
    vessel_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.trades_per_auction_range is None:
            self.trades_per_auction_range = [5, 7, 10]
        if self.vessel_counts is None:
            self.vessel_counts = {
                'suezmax': 1,
                'aframax': 1,
                'vlcc': 1
            }

class Tournament:
    def __init__(self, agents: List[TradingCompany], config: TournamentConfig):
        self.agents = agents
        self.config = config
        self.results = defaultdict(list)
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger('Tournament')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def process_simulation_results(self, metrics: Dict) -> Dict[str, Dict[str, float]]:
        """Process a single simulation's results into company statistics."""
        company_stats = {}
        
        for company_key in metrics["company_metrics"]:
            company_name = metrics["company_names"][company_key]
            
            # Calculate costs
            cost = metrics["company_metrics"][company_key].get("fuel_cost", 0)
            penalty = metrics["global_metrics"]["penalty"][company_key]
            
            # Calculate revenue
            all_outcomes = metrics["global_metrics"]["auction_outcomes"]
            all_outcomes_company_per_round = [d[company_key] for d in all_outcomes if company_key in d]
            all_outcomes_company = [x for sublist in all_outcomes_company_per_round for x in sublist]
            all_payments = [d["payment"] for d in all_outcomes_company]
            revenue = sum(all_payments)
            
            # Calculate final income and trades won
            income = revenue - cost - penalty
            trades_won = len(all_outcomes_company)
            
            company_stats[company_name] = {
                'total_profit': income,
                'revenue': revenue,
                'costs': cost,
                'penalty': penalty,
                'trades_won': trades_won,
                'average_profit_per_trade': income/trades_won if trades_won > 0 else 0
            }
            
        return company_stats
        
    def run_simulation(self, trades_per_auction: int, round_num: int) -> Dict[str, Any]:
        self.logger.info(f"Starting simulation round {round_num} with {trades_per_auction} trades per auction")
        
        try:
            # Create specification builder
            specs_builder = environment.get_specification_builder(
                environment_files_path="../../resources",
                trades_per_occurrence=trades_per_auction,
                num_auctions=self.config.num_months
            )
            
            # Add companies to simulation
            for agent_class in self.agents:
                fleet = fleets.mixed_fleet(
                    num_suezmax=self.config.vessel_counts['suezmax'],
                    num_aframax=self.config.vessel_counts['aframax'],
                    num_vlcc=self.config.vessel_counts['vlcc']
                )
                specs_builder.add_company(
                    agent_class.Data(agent_class, fleet, agent_class.__name__)
                )
                self.logger.info(f"Added agent: {agent_class.__name__}")
            
            # Run simulation
            sim = environment.generate_simulation(
                specs_builder,
                show_detailed_auction_outcome=True,
                global_agent_timeout=60
            )
            sim.run()
            
            # Extract results from JSON
            json_files = list(Path('.').glob('*.json'))
            if not json_files:
                self.logger.error("No JSON results file found")
                return {}
                
            self.logger.info(f"Found result file: {json_files[0]}")
            with open(json_files[0]) as f:
                metrics = json.load(f)
                
            # Process results
            results = {
                'company_statistics': self.process_simulation_results(metrics),
                'raw_metrics': metrics  # Keep raw data for potential additional analysis
            }
            
            # Clean up JSON file
            for json_file in json_files:
                os.remove(json_file)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {str(e)}")
            return {}
        
    def run_tournament(self):
        for trades_per_auction in self.config.trades_per_auction_range:
            self.logger.info(f"Starting tournament set with {trades_per_auction} trades per auction")
            round_results = []
            for round_num in range(self.config.num_rounds):
                results = self.run_simulation(trades_per_auction, round_num)
                if results and 'company_statistics' in results:
                    round_results.append(results)
                    self.logger.info(f"Completed round {round_num + 1}")
                else:
                    self.logger.warning(f"No valid results for round {round_num + 1}")
                    
            self.results[trades_per_auction] = round_results
                    
    def analyze_results(self) -> pd.DataFrame:
        data = []
        for trades_per_auction, rounds in self.results.items():
            for round_num, round_results in enumerate(rounds):
                for company, stats in round_results['company_statistics'].items():
                    data.append({
                        'trades_per_auction': trades_per_auction,
                        'round': round_num,
                        'company': company,
                        'total_profit': stats['total_profit'],
                        'trades_won': stats['trades_won'],
                        'average_profit_per_trade': stats['average_profit_per_trade'],
                        'revenue': stats['revenue'],
                        'costs': stats['costs'],
                        'penalty': stats['penalty']
                    })
        
        return pd.DataFrame(data)
        
    def plot_results(self):
        df = self.analyze_results()
        
        if df.empty:
            self.logger.error("No data available for plotting")
            return None
            
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
            
            # Plot 1: Total Profit by Company and Trades per Auction
            sns.boxplot(data=df, x='trades_per_auction', y='total_profit', 
                       hue='company', ax=ax1)
            ax1.set_title('Total Profit Distribution')
            ax1.set_xlabel('Trades per Auction')
            ax1.set_ylabel('Total Profit')
            
            # Plot 2: Trades Won by Company
            sns.barplot(data=df, x='company', y='trades_won', 
                       hue='trades_per_auction', ax=ax2)
            ax2.set_title('Trades Won by Company')
            ax2.set_xlabel('Company')
            ax2.set_ylabel('Number of Trades Won')
            
            # Plot 3: Revenue vs Costs
            df_melted = pd.melt(df, 
                              id_vars=['company', 'trades_per_auction', 'round'],
                              value_vars=['revenue', 'costs', 'penalty'],
                              var_name='metric', value_name='value')
            sns.boxplot(data=df_melted, x='company', y='value',
                       hue='metric', ax=ax3)
            ax3.set_title('Revenue, Costs, and Penalties by Company')
            ax3.set_xlabel('Company')
            ax3.set_ylabel('Value')
            
            # Plot 4: Average Profit per Trade
            sns.boxplot(data=df, x='trades_per_auction', y='average_profit_per_trade',
                       hue='company', ax=ax4)
            ax4.set_title('Average Profit per Trade')
            ax4.set_xlabel('Trades per Auction')
            ax4.set_ylabel('Average Profit per Trade')
            
            plt.tight_layout()
            plt.savefig('tournament_results.png')
            plt.close()
            
            # Create summary statistics
            summary = df.groupby(['company', 'trades_per_auction']).agg({
                'total_profit': ['mean', 'std'],
                'trades_won': ['mean', 'std'],
                'average_profit_per_trade': ['mean', 'std'],
                'revenue': ['mean', 'std'],
                'costs': ['mean', 'std'],
                'penalty': ['mean', 'std']
            }).round(2)
            
            summary.to_csv('tournament_summary.csv')
            
            # Print detailed tables for each company
            for company in df['company'].unique():
                company_data = df[df['company'] == company].mean()
                table = PrettyTable()
                table.field_names = ["Metric", "Value"]
                table.align["Value"] = "r"
                table.add_row(["Average Cost", round(company_data['costs'], 3)])
                table.add_row(["Average Penalty", round(company_data['penalty'], 3)])
                table.add_row(["Average Revenue", round(company_data['revenue'], 3)], divider=True)
                table.add_row(["Average Income", round(company_data['total_profit'], 3)])
                table.add_row(["Average Trades Won", round(company_data['trades_won'], 3)])
                print(f"\nCompany {company}")
                print(table)
                
            return summary
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
            return None

def main():
    # Example usage
    import sys
    sys.path.append('..')
    from MyCompany import MyCompany
    
    # Define agents to compete
    agents = [
        MyCompany,
        companies.MyArchEnemy,
        companies.TheScheduler
    ]
    
    # Configure tournament
    config = TournamentConfig(
        num_months=12,
        trades_per_auction_range=[5, 7, 10],
        num_rounds=1,
        vessel_counts={'suezmax': 1, 'aframax': 1, 'vlcc': 1}
    )
    
    # Create and run tournament
    tournament = Tournament(agents, config)
    tournament.run_tournament()
    
    # Analyze and visualize results
    summary = tournament.plot_results()
    if summary is not None:
        print("\nTournament Summary:")
        print(summary)
    else:
        print("\nNo results available to summarize")

if __name__ == "__main__":
    main()