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
    num_rounds: int = 1
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
        """Process a single simulation's results into detailed company statistics.
        
        Args:
            metrics (Dict): Raw metrics data from simulation output
            
        Returns:
            Dict[str, Dict[str, float]]: Processed statistics for each company
        """
        company_stats = {}
        
        try:
            # Process each company's metrics
            for company_key in metrics["company_metrics"]:
                company_name = metrics["company_names"].get(company_key, f"Unknown_Company_{company_key}")
                company_metrics = metrics["company_metrics"][company_key]
                
                # Extract vessel status metrics
                vessel_stats = {
                    key: value for key, value in company_metrics.items() 
                    if key.startswith('vessel_status_')
                }
                total_time = sum(vessel_stats.values())
                
                # Calculate vessel utilization percentages
                vessel_utilization = {
                    status.replace('vessel_status_', ''): (time / total_time * 100 if total_time > 0 else 0)
                    for status, time in vessel_stats.items()
                }
                
                # Calculate revenue from all auctions
                all_outcomes = metrics["global_metrics"]["auction_outcomes"]
                company_trades = []
                total_cargo_volume = 0
                
                for auction in all_outcomes:
                    if company_key in auction:
                        trades = auction[company_key]
                        company_trades.extend(trades)
                        # Sum up cargo volumes
                        total_cargo_volume += sum(trade["trade"]["amount"] for trade in trades)
                
                revenue = sum(trade["payment"] for trade in company_trades)
                trades_won = len(company_trades)
                
                # Calculate costs and penalties
                operational_costs = company_metrics.get("fuel_cost", 0)
                emissions = company_metrics.get("co2_emissions", 0)
                fuel_consumption = company_metrics.get("fuel_consumption", 0)
                penalty = metrics["global_metrics"]["penalty"][company_key]
                
                # Calculate final metrics
                total_costs = operational_costs + penalty
                net_income = revenue - total_costs
                
                company_stats[company_name] = {
                    # Financial metrics
                    'total_profit': net_income,
                    'revenue': revenue,
                    'operational_costs': operational_costs,
                    'penalty_costs': penalty,
                    'total_costs': total_costs,
                    'average_profit_per_trade': net_income/trades_won if trades_won > 0 else 0,
                    
                    # Operational metrics
                    'trades_won': trades_won,
                    'total_cargo_volume': total_cargo_volume,
                    'average_cargo_per_trade': total_cargo_volume/trades_won if trades_won > 0 else 0,
                    'fuel_consumption': fuel_consumption,
                    'co2_emissions': emissions,
                    'fuel_efficiency': total_cargo_volume/fuel_consumption if fuel_consumption > 0 else 0,
                    
                    # Vessel utilization metrics
                    **vessel_utilization,
                    
                    # Performance ratios
                    'revenue_per_unit_cargo': revenue/total_cargo_volume if total_cargo_volume > 0 else 0,
                    'cost_per_unit_cargo': total_costs/total_cargo_volume if total_cargo_volume > 0 else 0,
                    'profit_margin': (net_income/revenue * 100) if revenue > 0 else 0
                }
                
        except KeyError as e:
            self.logger.error(f"Missing expected key in metrics data: {e}")
            return {}
        except ZeroDivisionError as e:
            self.logger.error(f"Division by zero error in calculations: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error processing simulation results: {e}")
            return {}
            
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
                    # Get all the metrics we need, using .get() with defaults to avoid KeyError
                    entry = {
                        'trades_per_auction': trades_per_auction,
                        'round': round_num,
                        'company': company,
                        'total_profit': stats.get('total_profit', 0),
                        'trades_won': stats.get('trades_won', 0),
                        'revenue': stats.get('revenue', 0),
                        'operational_costs': stats.get('operational_costs', 0),
                        'penalty': stats.get('penalty_costs', 0),
                        'total_costs': stats.get('total_costs', 0),
                        'total_cargo_volume': stats.get('total_cargo_volume', 0),
                        'fuel_consumption': stats.get('fuel_consumption', 0),
                        'co2_emissions': stats.get('co2_emissions', 0),
                        'fuel_efficiency': stats.get('fuel_efficiency', 0),
                        'revenue_per_unit_cargo': stats.get('revenue_per_unit_cargo', 0),
                        'cost_per_unit_cargo': stats.get('cost_per_unit_cargo', 0),
                        'profit_margin': stats.get('profit_margin', 0),
                        'laden': stats.get('laden', 0),
                        'ballast': stats.get('ballast', 0),
                        'loading': stats.get('loading', 0),
                        'unloading': stats.get('unloading', 0),
                        'idle': stats.get('idle', 0)
                    }
                    
                    # Calculate derived metrics
                    trades = entry['trades_won'] or 1  # Avoid division by zero
                    entry['avg_cargo_per_trade'] = entry['total_cargo_volume'] / trades
                    
                    data.append(entry)
        
        return pd.DataFrame(data)
                
    def plot_results(self):
        """Generate comprehensive visualization and analysis of tournament results."""
        try:
            df = self.analyze_results()
            
            if df.empty:
                self.logger.error("No data available for plotting")
                return None
            
            # Create multiple figures for different aspects of analysis
            self._plot_financial_metrics(df)
            self._plot_operational_metrics(df)
            self._plot_efficiency_metrics(df)
            self._plot_vessel_utilization(df)
            
            # Generate and save detailed statistics
            self._generate_statistics(df)
            
            return self._create_summary(df)
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())  # Add detailed error traceback
            return None

    def _plot_financial_metrics(self, df: pd.DataFrame):
        """Create financial metrics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Total Profit by Company
        sns.boxplot(data=df, x='company', y='total_profit', ax=ax1)
        ax1.set_title('Total Profit Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Revenue Breakdown
        # Using the correct column names that exist in the DataFrame
        financial_metrics = ['revenue', 'operational_costs', 'penalty']
        df_melted = pd.melt(df, 
                        id_vars=['company'],
                        value_vars=financial_metrics,
                        var_name='metric', value_name='value')
        sns.barplot(data=df_melted, x='company', y='value', hue='metric', ax=ax2)
        ax2.set_title('Financial Breakdown by Company')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Profit Margins
        sns.boxplot(data=df, x='company', y='profit_margin', ax=ax3)
        ax3.set_title('Profit Margins by Company')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Cost per Unit Cargo
        sns.scatterplot(data=df, x='total_cargo_volume', y='cost_per_unit_cargo', 
                    hue='company', size='trades_won', ax=ax4)
        ax4.set_title('Cost Efficiency vs Cargo Volume')
        
        plt.tight_layout()
        plt.savefig('financial_metrics.png')
        plt.close()

    def _plot_operational_metrics(self, df: pd.DataFrame):
        """Create operational metrics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Trades Won Distribution
        sns.boxplot(data=df, x='company', y='trades_won', ax=ax1)
        ax1.set_title('Trades Won Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Cargo Volume Distribution
        sns.boxplot(data=df, x='company', y='total_cargo_volume', ax=ax2)
        ax2.set_title('Total Cargo Volume Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Average Cargo per Trade
        # Calculate average cargo per trade directly here
        df['avg_cargo_per_trade'] = df['total_cargo_volume'] / df['trades_won'].replace(0, 1)
        sns.boxplot(data=df, x='company', y='avg_cargo_per_trade', ax=ax3)
        ax3.set_title('Average Cargo per Trade')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Fuel Efficiency
        sns.scatterplot(data=df, x='fuel_consumption', y='total_cargo_volume', 
                    hue='company', size='trades_won', ax=ax4)
        ax4.set_title('Fuel Efficiency Analysis')
        
        plt.tight_layout()
        plt.savefig('operational_metrics.png')
        plt.close()

    def _plot_efficiency_metrics(self, df: pd.DataFrame):
        """Create efficiency metrics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: CO2 Emissions vs Cargo Volume
        sns.scatterplot(data=df, x='total_cargo_volume', y='co2_emissions',
                        hue='company', ax=ax1)
        ax1.set_title('Environmental Impact vs Cargo Volume')
        
        # Plot 2: Fuel Efficiency Distribution
        sns.boxplot(data=df, x='company', y='fuel_efficiency', ax=ax2)
        ax2.set_title('Fuel Efficiency Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Revenue per Unit Cargo
        sns.boxplot(data=df, x='company', y='revenue_per_unit_cargo', ax=ax3)
        ax3.set_title('Revenue per Unit Cargo')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Profit vs Vessel Utilization
        df['active_time_ratio'] = df[['laden', 'ballast']].sum(axis=1)
        sns.scatterplot(data=df, x='active_time_ratio', y='total_profit',
                        hue='company', size='trades_won', ax=ax4)
        ax4.set_title('Profit vs Vessel Activity')
        
        plt.tight_layout()
        plt.savefig('efficiency_metrics.png')
        plt.close()

    def _plot_vessel_utilization(self, df: pd.DataFrame):
        """Create vessel utilization visualization."""
        vessel_states = ['laden', 'ballast', 'loading', 'unloading', 'idle']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Vessel State Distribution
        df_melted = pd.melt(df, 
                            id_vars=['company'],
                            value_vars=vessel_states,
                            var_name='state', value_name='percentage')
        sns.boxplot(data=df_melted, x='company', y='percentage', 
                    hue='state', ax=ax1)
        ax1.set_title('Vessel State Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Active vs Idle Time
        df['active_time'] = df[['laden', 'ballast', 'loading', 'unloading']].sum(axis=1)
        df['idle_time'] = df['idle']
        time_data = pd.melt(df, 
                            id_vars=['company'],
                            value_vars=['active_time', 'idle_time'],
                            var_name='time_type', value_name='percentage')
        sns.barplot(data=time_data, x='company', y='percentage',
                    hue='time_type', ax=ax2)
        ax2.set_title('Active vs Idle Time')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('vessel_utilization.png')
        plt.close()

    # Also fix the correlations warning
    def _generate_statistics(self, df: pd.DataFrame):
        """Generate and save detailed statistics."""
        # Get numeric columns only
        metrics = [
            'total_profit', 'revenue', 'operational_costs', 'penalty',
            'trades_won', 'total_cargo_volume', 'fuel_consumption',
            'co2_emissions', 'fuel_efficiency', 'profit_margin'
        ]
        
        # Only include metrics that exist in the DataFrame
        available_metrics = [m for m in metrics if m in df.columns]
        
        summary = df.groupby('company')[available_metrics].agg(['mean', 'std', 'min', 'max'])
        summary.to_csv('detailed_summary.csv')
        
        # Fix the correlation analysis to avoid the deprecation warning
        correlations = df.groupby('company', group_keys=False).apply(
            lambda x: x[available_metrics].corr()
        )
        correlations.to_csv('metric_correlations.csv')

    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and display summary tables for each company."""
        summaries = {}
        
        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        for company in df['company'].unique():
            # Calculate means only for numeric columns
            company_data = df[df['company'] == company][numeric_columns].mean()
            
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            table.align["Value"] = "r"
            
            # Financial Performance
            table.add_row(["=== Financial Performance ===", ""])
            table.add_row(["Total Profit", f"{company_data['total_profit']:,.2f}"])
            table.add_row(["Revenue", f"{company_data['revenue']:,.2f}"])
            table.add_row(["Operational Costs", f"{company_data['operational_costs']:,.2f}"])
            table.add_row(["Penalty", f"{company_data['penalty']:,.2f}"])
            table.add_row(["Profit Margin", f"{company_data['profit_margin']:.2f}%"])
            
            # Operational Performance
            table.add_row(["=== Operational Performance ===", ""])
            table.add_row(["Trades Won", f"{company_data['trades_won']:.0f}"])
            table.add_row(["Total Cargo Volume", f"{company_data['total_cargo_volume']:,.2f}"])
            table.add_row(["Avg Cargo per Trade", f"{company_data['avg_cargo_per_trade']:,.2f}"])
            
            # Efficiency Metrics
            table.add_row(["=== Efficiency Metrics ===", ""])
            table.add_row(["Fuel Efficiency", f"{company_data['fuel_efficiency']:,.2f}"])
            table.add_row(["CO2 Emissions", f"{company_data['co2_emissions']:,.2f}"])
            table.add_row(["Revenue per Unit Cargo", f"{company_data['revenue_per_unit_cargo']:,.2f}"])
            
            # Vessel Utilization
            table.add_row(["=== Vessel Utilization (%) ===", ""])
            table.add_row(["Laden", f"{company_data['laden']:.1f}"])
            table.add_row(["Ballast", f"{company_data['ballast']:.1f}"])
            table.add_row(["Loading", f"{company_data['loading']:.1f}"])
            table.add_row(["Unloading", f"{company_data['unloading']:.1f}"])
            table.add_row(["Idle", f"{company_data['idle']:.1f}"])
            
            print(f"\nCompany: {company}")
            print(table)
            
            summaries[company] = company_data
        
        return pd.DataFrame(summaries)

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
        trades_per_auction_range=[5],
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