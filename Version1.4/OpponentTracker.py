from mable.shipping_market import AuctionLedger
from mable.shipping_market import Contract
from mable.cargo_bidding import TradingCompany
from typing import List

class OpponentTracker:
    def __init__(self, company: TradingCompany):
        self.company = company
        
    def receive(self, contracts: List[Contract], auction_ledger:AuctionLedger=None):
        pass 