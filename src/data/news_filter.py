import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time
import json

class NewsFilter:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger('NewsFilter')
        self.high_impact_events = []
        self.last_check = None
        
        # Economic calendar sources (you'll need to replace with actual API keys)
        self.news_sources = {
            'forexfactory': 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
        }
    
    def check_high_impact_news(self, symbols: List[str], 
                             lookahead_hours: int = 4) -> Dict[str, bool]:
        """Check for high-impact news events for given symbols"""
        try:
            current_time = datetime.now()
            
            # Only check once per hour to avoid API limits
            if (self.last_check and 
                (current_time - self.last_check) < timedelta(hours=1)):
                return self._get_cached_news_status(symbols)
            
            # Fetch economic calendar
            events = self._fetch_economic_calendar()
            if not events:
                return {symbol: False for symbol in symbols}
            
            # Filter high impact events for our symbols
            high_impact = self._filter_high_impact_events(events, symbols, lookahead_hours)
            self.high_impact_events = high_impact
            
            # Determine which symbols are affected
            affected_symbols = self._get_affected_symbols(high_impact, symbols)
            
            self.last_check = current_time
            self.logger.info(f"News check completed. High impact events: {len(high_impact)}")
            
            return affected_symbols
            
        except Exception as e:
            self.logger.error(f"Error checking news: {str(e)}")
            return {symbol: False for symbol in symbols}
    
    def _fetch_economic_calendar(self) -> List[Dict]:
        """Fetch economic calendar data"""
        try:
            response = requests.get(self.news_sources['forexfactory'], timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to fetch calendar: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching economic calendar: {str(e)}")
            return []
    
    def _filter_high_impact_events(self, events: List[Dict], 
                                 symbols: List[str], lookahead_hours: int) -> List[Dict]:
        """Filter high impact events for relevant symbols"""
        high_impact = []
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=lookahead_hours)
        
        for event in events:
            try:
                # Parse event time
                event_time = datetime.fromtimestamp(event['timestamp'])
                
                # Check if event is within our time window
                if not (current_time <= event_time <= end_time):
                    continue
                
                # Check impact level
                impact = event.get('impact', '').lower()
                if impact not in ['high', 'medium']:
                    continue
                
                # Check if event affects our symbols
                if self._event_affects_symbols(event, symbols):
                    high_impact.append(event)
                    
            except Exception as e:
                self.logger.warning(f"Error processing event: {str(e)}")
                continue
        
        return high_impact
    
    def _event_affects_symbols(self, event: Dict, symbols: List[str]) -> bool:
        """Check if event affects any of our symbols"""
        event_title = event.get('title', '').lower()
        event_currency = event.get('country', '').lower()
        
        # Map currencies to symbols
        currency_pairs = {
            'usd': ['USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD'],
            'eur': ['EURUSD', 'EURGBP', 'EURJPY'],
            'gbp': ['GBPUSD', 'EURGBP', 'GBPJPY'],
            'jpy': ['USDJPY', 'EURJPY', 'GBPJPY'],
            'aud': ['AUDUSD'],
            'cad': ['USDCAD'],
            'chf': ['USDCHF'],
            'gold': ['XAUUSD']
        }
        
        # Check if event currency affects our symbols
        for currency, pairs in currency_pairs.items():
            if currency in event_currency:
                for symbol in symbols:
                    if symbol in pairs:
                        return True
        
        # Check event title for specific keywords
        news_keywords = [
            'nfp', 'non-farm payrolls', 'fomc', 'fed', 'ecb', 'boe', 'boj',
            'cpi', 'inflation', 'interest rate', 'gdp', 'employment', 'retail sales'
        ]
        
        for keyword in news_keywords:
            if keyword in event_title:
                return True
        
        return False
    
    def _get_affected_symbols(self, events: List[Dict], symbols: List[str]) -> Dict[str, bool]:
        """Determine which symbols are affected by news events"""
        affected = {symbol: False for symbol in symbols}
        
        if not events:
            return affected
        
        # Map events to symbols
        for event in events:
            event_currency = event.get('country', '').lower()
            impact = event.get('impact', '').lower()
            
            # High impact events affect all related symbols
            if impact == 'high':
                if 'usd' in event_currency:
                    for symbol in ['USDJPY', 'EURUSD', 'GBPUSD']:
                        if symbol in symbols:
                            affected[symbol] = True
                elif 'eur' in event_currency:
                    for symbol in ['EURUSD', 'EURGBP']:
                        if symbol in symbols:
                            affected[symbol] = True
                elif 'jpy' in event_currency:
                    for symbol in ['USDJPY']:
                        if symbol in symbols:
                            affected[symbol] = True
                elif 'gold' in event_currency or 'xau' in event_currency:
                    for symbol in ['XAUUSD']:
                        if symbol in symbols:
                            affected[symbol] = True
        
        return affected
    
    def _get_cached_news_status(self, symbols: List[str]) -> Dict[str, bool]:
        """Get cached news status to avoid frequent API calls"""
        if not self.high_impact_events:
            return {symbol: False for symbol in symbols}
        
        return self._get_affected_symbols(self.high_impact_events, symbols)
    
    def should_avoid_trading(self, symbol: str, minutes_before: int = 60, 
                           minutes_after: int = 30) -> bool:
        """Check if we should avoid trading due to upcoming news"""
        try:
            events = self.check_high_impact_news([symbol])
            if not events.get(symbol, False):
                return False
            
            # Check if any high impact event is within the avoidance window
            current_time = datetime.now()
            
            for event in self.high_impact_events:
                event_time = datetime.fromtimestamp(event['timestamp'])
                time_diff = (event_time - current_time).total_seconds() / 60
                
                if -minutes_after <= time_diff <= minutes_before:
                    self.logger.info(f"Avoiding trading for {symbol} due to news: {event['title']}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading avoidance for {symbol}: {str(e)}")
            return False
    
    def get_upcoming_events(self, symbol: str, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming news events for a symbol"""
        events = self.check_high_impact_news([symbol])
        if not events.get(symbol, False):
            return []
        
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=hours_ahead)
        
        upcoming = []
        for event in self.high_impact_events:
            event_time = datetime.fromtimestamp(event['timestamp'])
            if current_time <= event_time <= end_time:
                upcoming.append(event)
        
        return upcoming