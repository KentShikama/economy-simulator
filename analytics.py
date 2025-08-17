import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
from typing import Dict, List, Any
import sys
import os
import json

# Add the economy-simulator-gui directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'economy-simulator-gui'))

from simulator import EconomySimulator

class EconomyAnalytics:
    def __init__(self, log_file: str = "economy_log.json"):
        self.action_log = self._load_from_file(log_file)
        self.df = self._create_dataframe()
    
    def _load_from_file(self, log_file: str) -> List[Dict[str, Any]]:
        """Load action log from line-based JSON file"""
        try:
            daily_logs = []
            with open(log_file, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    # Skip metadata line (first line)
                    if i == 0 and 'metadata' in data:
                        continue
                        
                    # This should be a daily log entry
                    if 'day' in data and 'actions' in data:
                        daily_logs.append(data)
                        
            return daily_logs
        except FileNotFoundError:
            print(f"Error: Log file '{log_file}' not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{log_file}': {e}")
            raise
        
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert action log to pandas DataFrame for easier analysis"""
        rows = []
        for day_entry in self.action_log:
            day = day_entry['day']
            for action in day_entry['actions']:
                row = {
                    'day': day,
                    'person': action['person'],
                    'city': action['city'],
                    'money': action['money'],
                    'fullness': action['fullness'],
                    'action': action['action'],
                    'seed_price': action['prices']['seed'],
                    'fertilizer_price': action['prices']['fertilizer'],
                    'grain_price': action['prices']['grain'],
                    'seed_inventory': action['inventory']['seed'],
                    'fertilizer_inventory': action['inventory']['fertilizer'],
                    'grain_inventory': action['inventory']['grain'],
                    'weights': action.get('weights', {})
                }
                rows.append(row)
        return pd.DataFrame(rows)
    
    def run_all_charts(self, output_dir: str = "charts"):
        """Generate individual chart files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Chart 1: Individual agent wealth over time
        plt.figure(figsize=(10, 6))
        self.chart_agent_wealth()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_agent_wealth.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 2: Inventory levels by agent
        plt.figure(figsize=(12, 10))
        self.chart_inventory_levels()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_inventory_by_agent.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 3: Resource prices by agent over time
        plt.figure(figsize=(12, 10))
        self.chart_prices_by_agent()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_prices_by_agent.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 4: Agent actions during crisis period
        plt.figure(figsize=(12, 10))
        self.chart_agent_crisis_actions()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_agent_crisis_actions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 5: Peddler travel record
        plt.figure(figsize=(12, 8))
        self.chart_peddler_travel()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_peddler_travel.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        
        # Chart 6: Market efficiency (price spreads)
        plt.figure(figsize=(10, 6))
        self.chart_market_efficiency()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/06_market_efficiency.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 7: Supply and demand balance
        plt.figure(figsize=(10, 8))
        self.chart_supply_demand()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/07_supply_demand.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chart 8: Agent decision weights over last 100 days
        plt.figure(figsize=(15, 12))
        self.chart_agent_weights()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/08_agent_weights.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"All charts saved to {output_dir}/ directory:")
    
    def chart_agent_wealth(self):
        """Chart 1: Individual agent wealth trajectories over time"""
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person].groupby('day')['money'].first()
            # Replace zero/negative values with small positive number for log scale
            money_values = person_data.values
            money_values = np.maximum(money_values, 0.01)  # Minimum value for log scale
            plt.plot(person_data.index, money_values, marker='o', markersize=3, linewidth=2, label=person)
        
        plt.title('Agent Wealth Over Time (Log Scale)', fontsize=12, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Money ($)')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    def chart_resource_prices(self):
        """Chart 2: Resource price trends over time"""
        price_data = self.df.groupby('day')[['seed_price', 'fertilizer_price', 'grain_price']].mean()
        
        plt.plot(price_data.index, price_data['seed_price'], marker='o', markersize=3, label='Seed', linewidth=2)
        plt.plot(price_data.index, price_data['fertilizer_price'], marker='s', markersize=3, label='Fertilizer', linewidth=2)
        plt.plot(price_data.index, price_data['grain_price'], marker='^', markersize=3, label='Grain', linewidth=2)
        
        plt.title('Average Resource Prices Over Time (Log Scale)', fontsize=12, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Price ($)')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    def chart_fullness_levels(self):
        """Chart 3: Agent fullness levels over time"""
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person].groupby('day')['fullness'].first()
            plt.plot(person_data.index, person_data.values, marker='o', markersize=3, linewidth=2, label=person, alpha=0.7)
        
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Danger Zone')
        plt.title('Agent Fullness Levels Over Time', fontsize=12, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Fullness (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
    
    def chart_trade_volume(self):
        """Chart 4: Daily trade value"""
        trade_actions = self.df[self.df['action'].str.contains('bought|sold', case=False, na=False)]
        
        # Calculate daily trade values
        daily_values = []
        days = []
        for day in sorted(trade_actions['day'].unique()):
            day_actions = trade_actions[trade_actions['day'] == day]['action']
            total_value = 0
            for action in day_actions:
                # Extract price from action text like "bought seed from X for 10.5"
                try:
                    if 'for ' in action:
                        price_str = action.split('for ')[-1].rstrip('.')
                        total_value += float(price_str)
                except:
                    pass
            daily_values.append(total_value)
            days.append(day)
        
        plt.plot(days, daily_values, marker='o', markersize=3, linewidth=2, color='green')
        plt.fill_between(days, daily_values, alpha=0.3, color='green')
        plt.title('Daily Trade Value (Log Scale)', fontsize=12, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Total Trade Value ($)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    def chart_inventory_levels(self):
        """Chart 5: Total inventory value per agent over time"""
        # Calculate total inventory value for each agent over time
        # Using a simple sum since we want to see overall inventory trends
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person].groupby('day').first()
            
            # Replace 0 values with small positive number for log scale
            seed_inventory = person_data['seed_inventory'].replace(0, 0.1)
            fertilizer_inventory = person_data['fertilizer_inventory'].replace(0, 0.1)
            grain_inventory = person_data['grain_inventory'].replace(0, 0.1)
            
            # Plot each resource type
            axes[0].plot(person_data.index, seed_inventory, marker='o', markersize=3, label=person, linewidth=2, alpha=0.7)
            axes[1].plot(person_data.index, fertilizer_inventory, marker='s', markersize=3, label=person, linewidth=2, alpha=0.7)
            axes[2].plot(person_data.index, grain_inventory, marker='^', markersize=3, label=person, linewidth=2, alpha=0.7)
        
        axes[0].set_ylabel('Seed Units')
        axes[0].set_title('Inventory Levels by Agent Over Time (Log Scale)', fontsize=12, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].set_ylim(bottom=0.1)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('Fertilizer Units')
        axes[1].set_yscale('log')
        axes[1].set_ylim(bottom=0.1)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel('Grain Units')
        axes[2].set_xlabel('Day')
        axes[2].set_yscale('log')
        axes[2].set_ylim(bottom=0.1)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[2].grid(True, alpha=0.3)
        
        plt.subplots_adjust(right=0.75)
    
    
    def chart_trade_relationships(self):
        """Chart 6: Trade relationship matrix - who trades with whom"""
        # Parse trade actions to find trading pairs
        trade_matrix = {}
        agents = self.df['person'].unique()
        
        for agent1 in agents:
            trade_matrix[agent1] = {agent2: 0 for agent2 in agents}
        
        # Count trades between agents
        for _, row in self.df.iterrows():
            action = row['action']
            person = row['person']
            
            if 'bought' in action or 'sold' in action:
                # Extract counterparty from action text
                if ' from ' in action:
                    counterparty = action.split(' from ')[-1].split(' for ')[0]
                elif ' to ' in action:
                    counterparty = action.split(' to ')[-1].split(' for ')[0]
                else:
                    continue
                    
                if counterparty in agents:
                    trade_matrix[person][counterparty] += 1
        
        # Create heatmap
        matrix_data = [[trade_matrix[a1][a2] for a2 in agents] for a1 in agents]
        
        plt.imshow(matrix_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Number of Trades')
        plt.xticks(range(len(agents)), agents, rotation=45, ha='right')
        plt.yticks(range(len(agents)), agents)
        plt.xlabel('Trading Partner')
        plt.ylabel('Agent')
        plt.title('Trade Relationships Matrix', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(agents)):
            for j in range(len(agents)):
                if matrix_data[i][j] > 0:
                    plt.text(j, i, str(matrix_data[i][j]), ha='center', va='center', color='white' if matrix_data[i][j] > max([max(row) for row in matrix_data])/2 else 'black')
    
    
    def chart_hunger_risk(self):
        """Chart 7: Hunger risk - minimum fullness levels reached"""
        # Track minimum fullness for each agent
        min_fullness = {}
        danger_days = {}  # Days when fullness < 30
        
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person]
            fullness_series = person_data.groupby('day')['fullness'].first()
            min_fullness[person] = fullness_series.min()
            danger_days[person] = (fullness_series < 30).sum()
        
        agents = list(min_fullness.keys())
        mins = list(min_fullness.values())
        dangers = list(danger_days.values())
        
        fig, ax1 = plt.subplots()
        
        colors = ['red' if m < 20 else 'orange' if m < 40 else 'green' for m in mins]
        bars = ax1.bar(agents, mins, color=colors, alpha=0.7)
        ax1.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Critical')
        ax1.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Warning')
        ax1.set_xlabel('Agent')
        ax1.set_ylabel('Minimum Fullness (%)', color='black')
        ax1.set_title('Hunger Risk Analysis', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        plt.xticks(rotation=45, ha='right')
        
        # Add danger day counts as text
        for i, (bar, danger) in enumerate(zip(bars, dangers)):
            if danger > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{danger}d', ha='center', va='bottom', fontsize=8)
        
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
    
    def chart_market_efficiency(self):
        """Chart 8: Market efficiency - price spreads between cities"""
        # Calculate price spreads for each resource using ratios (max/min)
        spreads_data = {'seed': [], 'fertilizer': [], 'grain': [], 'days': []}
        
        for day in sorted(self.df['day'].unique()):
            day_data = self.df[self.df['day'] == day]
            
            for resource in ['seed', 'fertilizer', 'grain']:
                price_col = f'{resource}_price'
                prices = day_data[price_col].values
                # Filter out zeros to avoid division by zero
                non_zero_prices = prices[prices > 0]
                
                if len(non_zero_prices) > 1:
                    # Use ratio (max/min) as the spread measure for log scale
                    min_price = non_zero_prices.min()
                    max_price = non_zero_prices.max()
                    
                    # Handle extreme ratios that cause overflow
                    if min_price == 0 or max_price / min_price > 1e100:
                        # Cap at 1e100 to prevent overflow while still showing extreme inefficiency
                        spread_ratio = 1e100
                    else:
                        spread_ratio = max_price / min_price
                    
                    spreads_data[resource].append(spread_ratio)
                else:
                    # If only one non-zero price or all zeros, perfect efficiency (ratio = 1)
                    spreads_data[resource].append(1.0)
            
            spreads_data['days'].append(day)
        
        plt.plot(spreads_data['days'], spreads_data['seed'], marker='o', markersize=3, label='Seed spread ratio', linewidth=2)
        plt.plot(spreads_data['days'], spreads_data['fertilizer'], marker='s', markersize=3, label='Fertilizer spread ratio', linewidth=2)
        plt.plot(spreads_data['days'], spreads_data['grain'], marker='^', markersize=3, label='Grain spread ratio', linewidth=2)
        
        plt.title('Price Spread Ratios (Market Inefficiency) - Log Scale', fontsize=12, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Price Spread Ratio (Max/Min)')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    
    def chart_agent_crisis_actions(self):
        """Chart 4: Agent actions during crisis period"""
        # Find agents who got dangerously low on fullness or had declining trends
        critical_agents = {}
        
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person].sort_values('day')
            
            # Find the day they hit minimum fullness
            min_fullness = person_data['fullness'].min()
            min_day_idx = person_data['fullness'].idxmin()
            min_day_row = person_data.loc[min_day_idx]
            
            # Also check if they had a severe decline in final 20 days
            if len(person_data) >= 20:
                recent_data = person_data.tail(20)
                fullness_decline = recent_data['fullness'].iloc[0] - recent_data['fullness'].iloc[-1]
            else:
                fullness_decline = 0
            
            # Include if they either hit very low fullness OR had major decline
            if min_fullness < 30 or fullness_decline > 30:
                # Get 10 days BEFORE their minimum fullness day
                min_day_num = min_day_row['day']
                before_min = person_data[person_data['day'] < min_day_num].tail(10)
                
                if len(before_min) == 0:  # If no data before min, use early days
                    analysis_period = person_data.head(10)
                else:
                    analysis_period = before_min
                
                # Count action types in analysis period - match actual action patterns
                action_counts = {
                    'consumed': len(analysis_period[analysis_period['action'].str.contains('consumed', case=False, na=False)]),
                    'bought': len(analysis_period[analysis_period['action'].str.contains('bought', case=False, na=False)]),
                    'sold': len(analysis_period[analysis_period['action'].str.contains('sold', case=False, na=False)]),
                    'refused/failed': len(analysis_period[analysis_period['action'].str.contains('refuses|cannot|out of stock|tried to', case=False, na=False)]),
                    'produced/collected': len(analysis_period[analysis_period['action'].str.contains('collected|produced|grew', case=False, na=False)]),
                    'resting/idle': len(analysis_period[analysis_period['action'].str.contains('resting|is resting', case=False, na=False)]),
                    'moving': len(analysis_period[analysis_period['action'].str.contains('moved|entered|left', case=False, na=False)])
                }
                
                # Calculate "other" as remaining actions
                total_categorized = sum(action_counts.values())
                action_counts['other'] = len(analysis_period) - total_categorized
                
                # Find who last traded with them in analysis period
                trade_actions = analysis_period[analysis_period['action'].str.contains('bought|sold', case=False, na=False)]
                last_trader = "No trades"
                if len(trade_actions) > 0:
                    last_action = trade_actions.iloc[-1]['action']
                    if ' from ' in last_action:
                        last_trader = last_action.split(' from ')[-1].split(' for ')[0]
                    elif ' to ' in last_action:
                        last_trader = last_action.split(' to ')[-1].split(' for ')[0]
                
                critical_agents[person] = {
                    'min_fullness': min_fullness,
                    'fullness_decline': fullness_decline,
                    'actions': action_counts,
                    'last_trader': last_trader,
                    'money_at_min': min_day_row['money'],
                    'grain_at_min': min_day_row['grain_inventory'],
                    'analysis_days': len(analysis_period)
                }
        
        if not critical_agents:
            plt.text(0.5, 0.5, 'No agents reached critical hunger levels', ha='center', va='center')
            plt.title('Agent Actions During Crisis Period', fontsize=12, fontweight='bold')
            return
        
        # Get the dying agent (lowest min fullness) to determine crisis period
        dying_agent = min(critical_agents.keys(), key=lambda x: critical_agents[x]['min_fullness'])
        dying_agent_data = self.df[self.df['person'] == dying_agent].sort_values('day')
        min_day_num = dying_agent_data.loc[dying_agent_data['fullness'].idxmin()]['day']
        
        # Get actions for ALL agents during the crisis period
        all_agents = list(self.df['person'].unique())
        comparison_actions = {}
        
        for agent in all_agents:
            agent_data = self.df[self.df['person'] == agent].sort_values('day')
            # Get same time period for all agents (50 days before the crisis)
            before_crisis = agent_data[agent_data['day'] < min_day_num].tail(50)
            
            action_counts = {
                'consumed': len(before_crisis[before_crisis['action'].str.contains('consumed', case=False, na=False)]),
                'bought': len(before_crisis[before_crisis['action'].str.contains('bought', case=False, na=False)]),
                'sold': len(before_crisis[before_crisis['action'].str.contains('sold', case=False, na=False)]),
                'refused/failed': len(before_crisis[before_crisis['action'].str.contains('refuses|cannot|out of stock|tried to', case=False, na=False)]),
                'produced/collected': len(before_crisis[before_crisis['action'].str.contains('collected|produced|grew', case=False, na=False)]),
                'resting/idle': len(before_crisis[before_crisis['action'].str.contains('resting|is resting', case=False, na=False)]),
                'moving': len(before_crisis[before_crisis['action'].str.contains('moved|entered|left', case=False, na=False)])
            }
            total_categorized = sum(action_counts.values())
            action_counts['other'] = len(before_crisis) - total_categorized
            comparison_actions[agent] = action_counts
        
        # Create single chart showing all agents
        action_types = ['consumed', 'bought', 'sold', 'refused/failed', 'produced/collected', 'resting/idle', 'moving', 'other']
        x = range(len(all_agents))
        width = 0.1
        
        for i, action in enumerate(action_types):
            values = [comparison_actions[agent][action] for agent in all_agents]
            plt.bar([xi + i*width for xi in x], values, width, label=action, alpha=0.7)
        
        plt.title(f'Agent Actions During Crisis Period (50 days before {dying_agent} hit {critical_agents[dying_agent]["min_fullness"]}% fullness)', fontsize=12, fontweight='bold')
        plt.ylabel('Action Count')
        plt.xticks([xi + width*3.5 for xi in x], all_agents, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    def chart_prices_by_agent(self):
        """Chart 11: Resource prices by agent over time"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        for person in self.df['person'].unique():
            person_data = self.df[self.df['person'] == person].groupby('day').first()
            
            # Replace 0 values with small positive number for log scale
            seed_prices = person_data['seed_price'].replace(0, 0.001)
            fertilizer_prices = person_data['fertilizer_price'].replace(0, 0.001)
            grain_prices = person_data['grain_price'].replace(0, 0.001)
            
            # Plot each resource price
            axes[0].plot(person_data.index, seed_prices, marker='o', markersize=3, label=person, linewidth=2, alpha=0.7)
            axes[1].plot(person_data.index, fertilizer_prices, marker='s', markersize=3, label=person, linewidth=2, alpha=0.7)
            axes[2].plot(person_data.index, grain_prices, marker='^', markersize=3, label=person, linewidth=2, alpha=0.7)
        
        axes[0].set_ylabel('Seed Price ($)')
        axes[0].set_title('Resource Prices by Agent Over Time (Log Scale)', fontsize=12, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('Fertilizer Price ($)')
        axes[1].set_yscale('log')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel('Grain Price ($)')
        axes[2].set_xlabel('Day')
        axes[2].set_yscale('log')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[2].grid(True, alpha=0.3)
        
        plt.subplots_adjust(right=0.75)
    
    def chart_peddler_travel(self):
        """Chart 5: Peddler travel record for last 100 days"""
        # Get peddlers (agents with 'Peddler' in their name)
        peddlers = [person for person in self.df['person'].unique() if 'Peddler' in person]
        
        if not peddlers:
            plt.text(0.5, 0.5, 'No peddlers found in data', ha='center', va='center')
            plt.title('Peddler Travel Record (Last 100 Days)', fontsize=12, fontweight='bold')
            return
        
        # Get last 100 days of data
        max_day = self.df['day'].max()
        start_day = max(0, max_day - 100)
        recent_data = self.df[self.df['day'] >= start_day]
        
        # Create subplots for each peddler
        n_peddlers = len(peddlers)
        fig, axes = plt.subplots(n_peddlers, 1, figsize=(12, 3 * n_peddlers), sharex=True)
        
        # If only one peddler, axes won't be a list
        if n_peddlers == 1:
            axes = [axes]
        
        cities = recent_data['city'].unique()
        colors = plt.cm.Set3(range(len(cities)))
        city_colors = dict(zip(cities, colors))
        
        for i, peddler in enumerate(peddlers):
            peddler_data = recent_data[recent_data['person'] == peddler].sort_values('day')
            
            if len(peddler_data) == 0:
                axes[i].text(0.5, 0.5, f'No data for {peddler}', ha='center', va='center')
                axes[i].set_title(f'{peddler} Travel Record', fontsize=10, fontweight='bold')
                continue
            
            # Track city changes and movements
            prev_city = None
            travel_events = []
            
            for _, row in peddler_data.iterrows():
                current_city = row['city']
                day = row['day']
                action = row['action']
                
                # Check for movement actions
                if 'moved' in action.lower() or 'entered' in action.lower() or 'left' in action.lower():
                    travel_events.append({
                        'day': day,
                        'city': current_city,
                        'action': action,
                        'type': 'movement'
                    })
                
                # Track city presence
                if current_city != prev_city:
                    travel_events.append({
                        'day': day,
                        'city': current_city,
                        'action': f'In {current_city}',
                        'type': 'location'
                    })
                    prev_city = current_city
            
            # Plot city presence as colored background
            for day in peddler_data['day'].unique():
                day_data = peddler_data[peddler_data['day'] == day]
                if len(day_data) > 0:
                    city = day_data.iloc[0]['city']
                    axes[i].axvspan(day-0.4, day+0.4, alpha=0.3, color=city_colors[city])
            
            # Plot movement events as black markers
            movement_days = [event['day'] for event in travel_events if event['type'] == 'movement']
            
            for day in movement_days:
                axes[i].scatter(day, 0.7, marker='>', s=50, color='black', edgecolor='white', linewidth=1, alpha=0.8)
            
            # Plot trade events with more detail
            trade_data = peddler_data[peddler_data['action'].str.contains('bought|sold', case=False, na=False)]
            
            # Different markers for different items
            item_markers = {'seed': 'o', 'fertilizer': 's', 'grain': '^'}
            
            for _, trade in trade_data.iterrows():
                action = trade['action'].lower()
                is_buy = 'bought' in action
                
                # Determine item type
                item = 'unknown'
                for item_name in ['seed', 'fertilizer', 'grain']:
                    if item_name in action:
                        item = item_name
                        break
                
                # Position based on buy/sell
                y_pos = 0.3 if is_buy else 0.1
                color = 'green' if is_buy else 'red'
                marker = item_markers.get(item, 'o')
                
                axes[i].scatter(trade['day'], y_pos, marker=marker, s=40, 
                              color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel(f'{peddler}', rotation=0, ha='right', va='center')
            axes[i].set_title(f'{peddler} Travel Record (Last 100 Days)', fontsize=10, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yticks([0.1, 0.3, 0.7])
            axes[i].set_yticklabels(['Sell', 'Buy', 'Move'])
        
        # Add legend for cities and trade types
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=city_colors[city], alpha=0.3, label=f'{city} (background)') for city in cities if city is not None]
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Buy Seed', linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8, label='Buy Fertilizer', linestyle='None'), 
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Buy Grain', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Sell Seed', linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='Sell Fertilizer', linestyle='None'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Sell Grain', linestyle='None'),
            plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='black', markersize=8, label='Movement', linestyle='None')
        ])
        
        axes[-1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[-1].set_xlabel('Day')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)
    
    def chart_supply_demand(self):
        """Chart 11: Supply and demand balance for grain"""
        days = sorted(self.df['day'].unique())
        
        grain_produced = []
        grain_consumed = []
        
        for day in days:
            day_data = self.df[self.df['day'] == day]
            
            # Count grain production
            produced = len(day_data[day_data['action'].str.contains('grew.*grain', case=False, na=False)]) * 10
            grain_produced.append(produced)
            
            # Count grain consumption
            consumed = len(day_data[day_data['action'].str.contains('consumed.*grain', case=False, na=False)])
            grain_consumed.append(consumed)
        
        # Calculate cumulative difference
        cumulative_balance = np.cumsum(np.array(grain_produced) - np.array(grain_consumed))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top: Production vs Consumption
        ax1.plot(days, grain_produced, marker='o', markersize=3, label='Grain Produced', linewidth=2, color='green')
        ax1.plot(days, grain_consumed, marker='s', markersize=3, label='Grain Consumed', linewidth=2, color='red')
        ax1.set_ylabel('Units per Day')
        ax1.set_title('Grain Supply vs Demand', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Cumulative balance
        ax2.plot(days, cumulative_balance, linewidth=2, color='blue')
        ax2.fill_between(days, cumulative_balance, alpha=0.3, color='blue')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Balance')
        ax2.set_title('Net Grain Balance Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def chart_agent_weights(self):
        """Chart 8: Agent decision weights over last 100 days"""
        # Get last 100 days of data
        if len(self.df) == 0:
            plt.text(0.5, 0.5, 'No data available for weight analysis', ha='center', va='center')
            plt.title('Agent Decision Weights (Last 100 Days)', fontsize=12, fontweight='bold')
            return
            
        max_day = self.df['day'].max()
        start_day = max(0, max_day - 100)
        recent_data = self.df[self.df['day'] >= start_day]
        
        # Get unique agents
        agents = recent_data['person'].unique()
        n_agents = len(agents)
        
        # Create a global color map for all possible actions
        all_possible_actions = set()
        for _, row in recent_data.iterrows():
            weights = row['weights']
            if isinstance(weights, dict) and weights:
                all_possible_actions.update(weights.keys())
        
        all_possible_actions = sorted(list(all_possible_actions))
        action_colors = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_possible_actions)))
        for i, action in enumerate(all_possible_actions):
            action_colors[action] = colors[i]
        
        # Create subplots for each agent
        fig, axes = plt.subplots(n_agents, 1, figsize=(15, 3 * n_agents), sharex=True)
        if n_agents == 1:
            axes = [axes]
        
        for i, agent in enumerate(agents):
            agent_data = recent_data[recent_data['person'] == agent].sort_values('day')
            
            if len(agent_data) == 0:
                axes[i].text(0.5, 0.5, f'No data for {agent}', ha='center', va='center')
                axes[i].set_title(f'{agent} Decision Weights', fontsize=10, fontweight='bold')
                continue
            
            # Extract weights data for this agent
            weights_by_day = {}
            all_actions = set()
            
            for _, row in agent_data.iterrows():
                day = row['day']
                weights = row['weights']
                if isinstance(weights, dict) and weights:
                    weights_by_day[day] = weights
                    all_actions.update(weights.keys())
            
            if not weights_by_day:
                axes[i].text(0.5, 0.5, f'No weights data for {agent}', ha='center', va='center')
                axes[i].set_title(f'{agent} Decision Weights', fontsize=10, fontweight='bold')
                continue
            
            # Convert to consistent format
            days = sorted(weights_by_day.keys())
            all_actions = sorted(list(all_actions))
            
            # Create stacked area chart
            weight_matrix = []
            for day in days:
                day_weights = weights_by_day[day]
                total_weight = sum(day_weights.values()) if day_weights.values() else 1
                # Normalize weights to percentages
                normalized_weights = []
                for action in all_actions:
                    weight = day_weights.get(action, 0)
                    normalized_weights.append(weight / total_weight * 100 if total_weight > 0 else 0)
                weight_matrix.append(normalized_weights)
            
            # Transpose for plotting
            weight_matrix = np.array(weight_matrix).T
            
            # Create stacked area plot
            bottom = np.zeros(len(days))
            
            for j, action in enumerate(all_actions):
                axes[i].fill_between(days, bottom, bottom + weight_matrix[j], 
                                   alpha=0.7, label=action, color=action_colors[action])
                bottom += weight_matrix[j]
            
            axes[i].set_ylabel('Weight %')
            axes[i].set_title(f'{agent} Decision Weights (Last 100 Days)', fontsize=10, fontweight='bold')
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 100)
        
        axes[-1].set_xlabel('Day')
        plt.suptitle('Agent Decision Weight Evolution', fontsize=14, fontweight='bold', y=0.995)
        plt.subplots_adjust(right=0.8, hspace=0.3)


def main():
    """Main function to analyze economy_log.json and generate charts"""
    print("Economy Simulator Analytics")
    print("=" * 50)
    print("Loading data from economy_log.json...")
    
    try:
        analytics = EconomyAnalytics()
        print(f"Loaded {len(analytics.df)} actions from {len(analytics.action_log)} days")
        
        analytics.run_all_charts("charts")
        
        # Print summary statistics
        print("\n=== ANALYSIS SUMMARY ===")
        if len(analytics.df) > 0:
            print(f"Total days in log: {analytics.df['day'].max()}")
            print(f"Total actions logged: {len(analytics.df)}")
            
            final_day_data = analytics.df[analytics.df['day'] == analytics.df['day'].max()]
            print("\nFinal agent status:")
            for person in final_day_data['person'].unique():
                person_data = final_day_data[final_day_data['person'] == person].iloc[0]
                print(f"  {person}: ${person_data['money']:.2f}, {person_data['fullness']}% fullness")
        else:
            print("No data found in log file")
        
        print(f"\nCharts generated in charts/ directory:")
        print("  01_agent_wealth.png - Agent Wealth Over Time")
        print("  02_inventory_by_agent.png - Inventory by Agent Over Time")
        print("  03_prices_by_agent.png - Resource Prices by Agent Over Time")
        print("  04_agent_crisis_actions.png - Agent Actions During Crisis Period")
        print("  05_peddler_travel.png - Peddler Travel Record (Last 100 Days)")
        print("  06_market_efficiency.png - Price Spreads (Market Inefficiency)")
        print("  07_supply_demand.png - Grain Supply vs Demand Balance")
        print("  08_agent_weights.png - Agent Decision Weights (Last 100 Days)")
        
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        print("Make sure economy_log.json exists by running the simulator first!")


if __name__ == "__main__":
    main()