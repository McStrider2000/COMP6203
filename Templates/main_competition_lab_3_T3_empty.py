from mable.cargo_bidding import TradingCompany
from mable.examples import environment, fleets, shipping, companies


class MyCompany(TradingCompany):

    def inform(self, trades, *args, **kwargs):
        return []

    def find_closest_vessel_for_company(self, trade, company):
        """Helper function to find closest vessel for a specific company"""
        closest_vessel = None
        min_distance = float('inf')
        
        for vessel in company._fleet:
            distance = self.headquarters.get_network_distance(
                vessel.location, 
                trade.origin_port
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_vessel = vessel
        
        return closest_vessel

    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        competitor_name = "Arch Enemy Ltd."
        competitor_won_contracts = auction_ledger[competitor_name]
        competitor_fleet = [c for c in self.headquarters.get_companies() 
                        if c.name == competitor_name]
        
        if not competitor_fleet:  # Safety check
            print(f"Warning: Competitor {competitor_name} not found")
            return
            
        competitor = competitor_fleet.pop()
        
        print("\nAnalyzing Arch Enemy Ltd.'s bidding behavior:")
        print("-" * 50)
        
        for contract in competitor_won_contracts:
            trade = contract.trade
            winning_price = contract.payment
            
            # Find closest vessel for the winning company only
            likely_vessel = self.find_closest_vessel_for_company(trade, competitor)
            
            if likely_vessel:
                # Calculate our prediction of their base cost
                predicted_cost = self.predict_cost(likely_vessel, trade)
                
                # Calculate their profit factor (winning price / predicted cost)
                profit_factor = winning_price / predicted_cost if predicted_cost > 0 else 0
                
                print(f"\nTrade: {trade.origin_port.name} -> {trade.destination_port.name}")
                print(f"Likely vessel used: {likely_vessel.name}")
                print(f"Winning price: {winning_price:.2f}")
                print(f"Predicted cost: {predicted_cost:.2f}")
                print(f"Implied profit factor: {profit_factor:.2f}")
            else:
                print(f"\nWarning: Could not find a likely vessel for trade from {trade.origin_port.name} to {trade.destination_port.name}")


    def predict_cost(self, vessel, trade):
        # Basic cost calculation based on loading, unloading, and travel
        time_to_load = vessel.get_loading_time(trade.cargo_type, trade.amount)
        
        # Get travel distance between origin and destination
        travel_distance = self.headquarters.get_network_distance(
            trade.origin_port, 
            trade.destination_port
        )
        
        loading_cost = vessel.get_loading_consumption(time_to_load)

        time_to_pick_up = vessel.get_travel_time(travel_distance)

        travelling_cost = vessel.get_laden_consumption(time_to_pick_up, vessel.speed)
        
        # Calculate total cost based on vessel's hourly cost
        total_cost = loading_cost + loading_cost  + travelling_cost
        
        return total_cost


def build_specification():
    specifications_builder = environment.get_specification_builder(fixed_trades=shipping.example_trades_1())
    fleet = fleets.example_fleet_1()
    specifications_builder.add_company(MyCompany.Data(MyCompany, fleet, MyCompany.__name__))
    specifications_builder.add_company(
        companies.MyArchEnemy.Data(companies.MyArchEnemy, fleets.example_fleet_2(), "Arch Enemy Ltd."))
    sim = environment.generate_simulation(
        specifications_builder,
        show_detailed_auction_outcome=True)
    sim.run()


if __name__ == '__main__':
    build_specification()
