#!/usr/bin/env python3
"""Analyze captured battle replays from entity model benchmarks."""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class BattleStats:
    game_number: int
    outcome: str  # 'win' or 'loss'
    turns: int
    rl_switches: int
    random_switches: int
    rl_pokemon_fainted: int
    random_pokemon_fainted: int
    critical_hits: int
    super_effective: int
    resisted: int

def extract_battle_log(html_path):
    """Extract battle log from HTML replay file."""
    with open(html_path, 'r') as f:
        content = f.read()
    
    # Find the script tag with battle log data
    match = re.search(r'<script[^>]*class="battle-log-data">(.*?)</script>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def parse_battle_log(log_text):
    """Parse battle log and extract statistics."""
    lines = log_text.split('\n')
    
    stats = {
        'turns': 0,
        'rl_switches': 0,
        'random_switches': 0,
        'rl_pokemon_fainted': 0,
        'random_pokemon_fainted': 0,
        'critical_hits': 0,
        'super_effective': 0,
        'resisted': 0,
        'rl_side': None,
        'random_side': None,
        'rl_fainted_count': 0,
        'random_fainted_count': 0,
    }
    
    for line in lines:
        parts = line.split('|')
        
        # Track sides
        if parts[0] == 'player':
            if 'RandomBot' in line:
                stats['random_side'] = parts[2]
            elif 'RLBot' in line:
                stats['rl_side'] = parts[2]
        
        # Count turns
        if parts[0] == 'turn':
            stats['turns'] = int(parts[1])
        
        # Count switches
        if parts[0] == 'switch':
            if 'p1' in parts[1]:
                stats['random_switches'] += 1
            else:
                stats['rl_switches'] += 1
        
        # Count faints
        if parts[0] == 'faint':
            if 'p1' in parts[1]:
                stats['random_fainted_count'] += 1
            else:
                stats['rl_fainted_count'] += 1
        
        # Count effects
        if parts[0] == '-crit':
            stats['critical_hits'] += 1
        elif parts[0] == '-supereffective':
            stats['super_effective'] += 1
        elif parts[0] == '-resisted':
            stats['resisted'] += 1
    
    return stats

def analyze_replays(replay_dir):
    """Analyze all replays in a directory."""
    replay_dir = Path(replay_dir)
    replays = sorted(replay_dir.glob('*.html'))
    
    win_stats = []
    loss_stats = []
    
    print(f"\nAnalyzing {len(replays)} replays from {replay_dir}")
    print("=" * 80)
    
    for replay_path in replays:
        filename = replay_path.name
        
        # Determine outcome from filename
        if '-win-' in filename:
            outcome = 'win'
        elif '-loss-' in filename:
            outcome = 'loss'
        else:
            continue
        
        # Extract game number
        match = re.search(r'game-(\d+)', filename)
        game_number = int(match.group(1)) if match else 0
        
        # Extract and parse log
        battle_log = extract_battle_log(replay_path)
        stats = parse_battle_log(battle_log)
        
        # Create battle stats object
        battle = BattleStats(
            game_number=game_number,
            outcome=outcome,
            turns=stats['turns'],
            rl_switches=stats['rl_switches'],
            random_switches=stats['random_switches'],
            rl_pokemon_fainted=stats['rl_fainted_count'],
            random_pokemon_fainted=stats['random_fainted_count'],
            critical_hits=stats['critical_hits'],
            super_effective=stats['super_effective'],
            resisted=stats['resisted'],
        )
        
        if outcome == 'win':
            win_stats.append(battle)
        else:
            loss_stats.append(battle)
    
    return win_stats, loss_stats

def print_analysis(win_stats, loss_stats):
    """Print diagnostic analysis."""
    print("\n" + "="*80)
    print("BATTLE ANALYSIS SUMMARY")
    print("="*80)
    
    total_games = len(win_stats) + len(loss_stats)
    win_rate = len(win_stats) / total_games * 100 if total_games > 0 else 0
    
    print(f"\nWin Rate: {len(win_stats)}/{total_games} ({win_rate:.1f}%)")
    
    if win_stats:
        print(f"\n--- WINS ({len(win_stats)} games) ---")
        avg_turns = sum(w.turns for w in win_stats) / len(win_stats)
        avg_switches = sum(w.rl_switches for w in win_stats) / len(win_stats)
        avg_opponent_switches = sum(w.random_switches for w in win_stats) / len(win_stats)
        
        print(f"Avg Turns: {avg_turns:.1f}")
        print(f"Avg RL Switches: {avg_switches:.1f}")
        print(f"Avg Random Switches: {avg_opponent_switches:.1f}")
        print(f"Total Critical Hits: {sum(w.critical_hits for w in win_stats)}")
        print(f"Total Super Effective: {sum(w.super_effective for w in win_stats)}")
        print(f"Avg Random Pokémon Fainted: {sum(w.random_pokemon_fainted for w in win_stats) / len(win_stats):.1f}")
    
    if loss_stats:
        print(f"\n--- LOSSES ({len(loss_stats)} games) ---")
        avg_turns = sum(l.turns for l in loss_stats) / len(loss_stats)
        avg_switches = sum(l.rl_switches for l in loss_stats) / len(loss_stats)
        avg_opponent_switches = sum(l.random_switches for l in loss_stats) / len(loss_stats)
        
        print(f"Avg Turns: {avg_turns:.1f}")
        print(f"Avg RL Switches: {avg_switches:.1f}")
        print(f"Avg Random Switches: {avg_opponent_switches:.1f}")
        print(f"Total Critical Hits: {sum(l.critical_hits for l in loss_stats)}")
        print(f"Total Super Effective: {sum(l.super_effective for l in loss_stats)}")
        print(f"Avg RL Pokémon Fainted: {sum(l.rl_pokemon_fainted for l in loss_stats) / len(loss_stats):.1f}")
    
    # Comparison
    print(f"\n--- COMPARISON ---")
    if win_stats and loss_stats:
        win_avg_switches = sum(w.rl_switches for w in win_stats) / len(win_stats)
        loss_avg_switches = sum(l.rl_switches for l in loss_stats) / len(loss_stats)
        print(f"RL Switches in Wins: {win_avg_switches:.1f} vs Losses: {loss_avg_switches:.1f}")
        
        win_avg_opp_switches = sum(w.random_switches for w in win_stats) / len(win_stats)
        loss_avg_opp_switches = sum(l.random_switches for l in loss_stats) / len(loss_stats)
        print(f"Random Switches in Wins: {win_avg_opp_switches:.1f} vs Losses: {loss_avg_opp_switches:.1f}")
        
        win_avg_turns = sum(w.turns for w in win_stats) / len(win_stats)
        loss_avg_turns = sum(l.turns for l in loss_stats) / len(loss_stats)
        print(f"Avg Battle Duration (wins): {win_avg_turns:.1f} turns vs (losses): {loss_avg_turns:.1f} turns")

if __name__ == '__main__':
    replay_dir = Path('pokemon-showdown-model-feature/logs/replays')
    win_stats, loss_stats = analyze_replays(replay_dir)
    print_analysis(win_stats, loss_stats)
    
    # Also save summary
    summary = {
        'total_games': len(win_stats) + len(loss_stats),
        'wins': len(win_stats),
        'losses': len(loss_stats),
        'win_rate': len(win_stats) / (len(win_stats) + len(loss_stats)) * 100,
    }
    
    print("\n" + "="*80)
    print(f"Summary saved: {summary}")

