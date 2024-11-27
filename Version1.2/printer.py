class PrettyPrinter:
    @staticmethod
    def print_schedule_analysis(trades, tradesToIdxs, cost_comparisons):
        PrettyPrinter.print_schedule_details(trades, tradesToIdxs, cost_comparisons)
        if cost_comparisons:
            PrettyPrinter.print_aggregate_statistics(cost_comparisons)

    @staticmethod
    def print_schedule_details(trades, tradesToIdxs, cost_comparisons):
        print("\nSchedule and Cost Analysis:")
        print("-" * 80)
        
        for trade in trades:
            if trade in tradesToIdxs:
                PrettyPrinter.print_scheduled_trade_details(trade, tradesToIdxs, cost_comparisons)
            else:
                PrettyPrinter.print_unscheduled_trade_details(trade)

    @staticmethod
    def print_scheduled_trade_details(trade, tradesToIdxs, cost_comparisons):
        print(f"\nTrade: {trade.origin_port} -> {trade.destination_port}")
        print(f"Schedule: Start at idx {tradesToIdxs[trade][0]}, End at idx {tradesToIdxs[trade][1]}")
        
        if trade in cost_comparisons:
            PrettyPrinter.print_cost_comparison(cost_comparisons[trade])

    @staticmethod
    def print_unscheduled_trade_details(trade):
        print(f"\nTrade: {trade.origin_port} -> {trade.destination_port} (Could not be scheduled)")

    @staticmethod
    def print_cost_comparison(comparison):
        print("Cost Analysis:")
        print(f"  Detailed cost (find_cheapest_schedule)   : {comparison['detailed_cost']:.2f}")
        print(f"  Estimated cost (estimate_fulfilment_cost): {comparison['estimated_cost']:.2f}")
        print(f"  Difference: {comparison['difference']:.2f}")
        print(f"  Difference percentage: {comparison['difference_percentage']:.2f}%")

    @staticmethod
    def print_aggregate_statistics(cost_comparisons):
        print("\nAggregate Cost Analysis:")
        print("-" * 80)
        
        statistics = PrettyPrinter.calculate_aggregate_statistics(cost_comparisons)
        
        print(f"Average cost difference: {statistics['avg_difference']:.2f}")
        print(f"Average difference percentage: {statistics['avg_difference_percentage']:.2f}%")
        print(f"Maximum difference: {statistics['max_difference']:.2f}")
        print(f"Minimum difference: {statistics['min_difference']:.2f}")

    @staticmethod
    def calculate_aggregate_statistics(cost_comparisons):
        differences = [c['difference'] for c in cost_comparisons.values()]
        difference_percentages = [c['difference_percentage'] for c in cost_comparisons.values()]
        
        return {
            'avg_difference': sum(differences) / len(differences),
            'avg_difference_percentage': sum(difference_percentages) / len(difference_percentages),
            'max_difference': max(differences),
            'min_difference': min(differences)
        }
    

    def print_fleet_schedules(fleet):
        for vessel in fleet:
            print("------------------------------------------")
            print("Vessel Name", vessel.name)
            print("Locations schedule of vessel",vessel.schedule._get_node_locations())
            print("Insertion points of vessel",vessel.schedule.get_insertion_points())
            print("------------------------------------------")


def print_lowest_long_to_short(lowest_cost_increase, orginal_lowest_cost_increase):
    """Print cost increase comparison in red text."""
    RED = '\033[91m'    # Red text
    ENDC = '\033[0m'    # Reset to default color
    
    print(f"{RED}Lowest cost increase from long to short schedule{ENDC}")
    print(f"{RED}Cost increased from {lowest_cost_increase} to {orginal_lowest_cost_increase}{ENDC}")