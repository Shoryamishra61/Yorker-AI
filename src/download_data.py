"""
IPL Data Download & Preparation Script
Downloads historical IPL match data from Cricsheet and prepares it for ML pipeline.
"""
import os
import requests
import zipfile
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── IPL 2026 Season Data (hardcoded from PRD - latest as of April 22, 2026) ───

IPL_2026_TEAMS = {
    'RCB': 'Royal Challengers Bengaluru',
    'RR': 'Rajasthan Royals',
    'MI': 'Mumbai Indians',
    'CSK': 'Chennai Super Kings',
    'SRH': 'Sunrisers Hyderabad',
    'GT': 'Gujarat Titans',
    'PBKS': 'Punjab Kings',
    'DC': 'Delhi Capitals',
    'KKR': 'Kolkata Knight Riders',
    'LSG': 'Lucknow Super Giants'
}

POINTS_TABLE_2026 = [
    {'team': 'RCB', 'matches': 5, 'wins': 4, 'losses': 1, 'nr': 0, 'points': 8, 'nrr': 1.503},
    {'team': 'RR', 'matches': 7, 'wins': 5, 'losses': 2, 'nr': 0, 'points': 10, 'nrr': 0.8},
    {'team': 'PBKS', 'matches': 6, 'wins': 4, 'losses': 2, 'nr': 0, 'points': 8, 'nrr': 0.5},
    {'team': 'SRH', 'matches': 7, 'wins': 3, 'losses': 4, 'nr': 0, 'points': 6, 'nrr': 0.2},
    {'team': 'CSK', 'matches': 6, 'wins': 3, 'losses': 3, 'nr': 0, 'points': 6, 'nrr': 0.1},
    {'team': 'GT', 'matches': 7, 'wins': 3, 'losses': 4, 'nr': 0, 'points': 6, 'nrr': -0.3},
    {'team': 'MI', 'matches': 7, 'wins': 3, 'losses': 4, 'nr': 0, 'points': 6, 'nrr': -0.4},
    {'team': 'DC', 'matches': 6, 'wins': 3, 'losses': 3, 'nr': 0, 'points': 6, 'nrr': -0.2},
    {'team': 'LSG', 'matches': 7, 'wins': 2, 'losses': 5, 'nr': 0, 'points': 4, 'nrr': -0.9},
    {'team': 'KKR', 'matches': 6, 'wins': 2, 'losses': 4, 'nr': 0, 'points': 4, 'nrr': -1.1},
]

BATTING_LEADERS_2026 = [
    {'player': 'Abhishek Sharma', 'team': 'SRH', 'matches': 7, 'runs': 323, 'avg': 46.1, 'sr': 178.0, 'top_score': 90},
    {'player': 'Heinrich Klaasen', 'team': 'SRH', 'matches': 7, 'runs': 305, 'avg': 50.8, 'sr': 185.0, 'top_score': 85},
    {'player': 'Vaibhav Suryavanshi', 'team': 'RR', 'matches': 7, 'runs': 285, 'avg': 40.7, 'sr': 175.0, 'top_score': 78},
    {'player': 'Sanju Samson', 'team': 'CSK', 'matches': 6, 'runs': 265, 'avg': 52.0, 'sr': 185.0, 'top_score': 115},
    {'player': 'Tilak Varma', 'team': 'MI', 'matches': 7, 'runs': 255, 'avg': 41.0, 'sr': 172.0, 'top_score': 101},
    {'player': 'Shubman Gill', 'team': 'GT', 'matches': 7, 'runs': 245, 'avg': 34.0, 'sr': 145.0, 'top_score': 75},
    {'player': 'Virat Kohli', 'team': 'RCB', 'matches': 5, 'runs': 210, 'avg': 50.0, 'sr': 152.0, 'top_score': 85},
]

BOWLING_LEADERS_2026 = [
    {'player': 'Anshul Kamboj', 'team': 'CSK', 'matches': 6, 'wickets': 13, 'economy': 9.74, 'best': '3/22'},
    {'player': 'Prasidh Krishna', 'team': 'GT', 'matches': 5, 'wickets': 12, 'economy': 9.20, 'best': '4/28'},
    {'player': 'Prince Yadav', 'team': 'LSG', 'matches': 6, 'wickets': 11, 'economy': 8.59, 'best': '3/30'},
    {'player': 'Bhuvneshwar Kumar', 'team': 'RCB', 'matches': 6, 'wickets': 10, 'economy': 8.33, 'best': '3/25'},
    {'player': 'Kagiso Rabada', 'team': 'GT', 'matches': 7, 'wickets': 10, 'economy': 8.80, 'best': '3/31'},
]

VENUE_DATA = [
    {'venue': 'M. Chinnaswamy Stadium', 'city': 'Bengaluru', 'pitch': 'batting', 'avg_1st': 185, 'chase_pct': 52, 'home': 'RCB'},
    {'venue': 'M.A. Chidambaram Stadium', 'city': 'Chennai', 'pitch': 'spin', 'avg_1st': 161, 'chase_pct': 44, 'home': 'CSK'},
    {'venue': 'Wankhede Stadium', 'city': 'Mumbai', 'pitch': 'balanced', 'avg_1st': 178, 'chase_pct': 54, 'home': 'MI'},
    {'venue': 'Eden Gardens', 'city': 'Kolkata', 'pitch': 'batting', 'avg_1st': 182, 'chase_pct': 55, 'home': 'KKR'},
    {'venue': 'Narendra Modi Stadium', 'city': 'Ahmedabad', 'pitch': 'balanced', 'avg_1st': 175, 'chase_pct': 51, 'home': 'GT'},
    {'venue': 'Arun Jaitley Stadium', 'city': 'Delhi', 'pitch': 'batting', 'avg_1st': 188, 'chase_pct': 56, 'home': 'DC'},
    {'venue': 'Rajiv Gandhi Intl Stadium', 'city': 'Hyderabad', 'pitch': 'batting', 'avg_1st': 190, 'chase_pct': 58, 'home': 'SRH'},
    {'venue': 'HPCA Cricket Stadium', 'city': 'Dharamsala', 'pitch': 'pace', 'avg_1st': 170, 'chase_pct': 50, 'home': 'PBKS'},
    {'venue': 'Maharaja YS Intl Stadium', 'city': 'New Chandigarh', 'pitch': 'balanced', 'avg_1st': 176, 'chase_pct': 53, 'home': 'PBKS'},
    {'venue': 'Ekana Cricket Stadium', 'city': 'Lucknow', 'pitch': 'balanced', 'avg_1st': 172, 'chase_pct': 52, 'home': 'LSG'},
    {'venue': 'Sawai Mansingh Stadium', 'city': 'Jaipur', 'pitch': 'balanced', 'avg_1st': 174, 'chase_pct': 52, 'home': 'RR'},
]


def download_cricsheet_data():
    """Download IPL match data from Cricsheet."""
    url = "https://cricsheet.org/downloads/ipl_csv2.zip"
    zip_path = os.path.join(RAW_DIR, "ipl_csv2.zip")
    
    if os.path.exists(os.path.join(RAW_DIR, 'ipl_csv2')):
        print("Cricsheet data already downloaded.")
        return True
    
    print(f"Downloading Cricsheet IPL data from {url}...")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {os.path.getsize(zip_path) / 1e6:.1f} MB")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(os.path.join(RAW_DIR, 'ipl_csv2'))
        print("Extracted successfully.")
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("Will generate synthetic historical data for model training.")
        return False


def generate_historical_data():
    """Generate comprehensive historical IPL match data for training (2008-2025)."""
    np.random.seed(42)
    
    teams_by_season = {
        2008: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2009: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2010: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2011: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2012: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2013: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2014: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2015: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2016: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2017: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2018: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2019: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2020: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2021: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH'],
        2022: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH', 'GT', 'LSG'],
        2023: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH', 'GT', 'LSG'],
        2024: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH', 'GT', 'LSG'],
        2025: ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH', 'GT', 'LSG'],
    }
    
    # Team strength priors (historical win rates roughly)
    team_strength = {
        'MI': 0.58, 'CSK': 0.60, 'KKR': 0.50, 'RCB': 0.48, 'RR': 0.47,
        'DC': 0.45, 'PBKS': 0.43, 'SRH': 0.52, 'GT': 0.55, 'LSG': 0.48
    }
    
    venues = [v['venue'] for v in VENUE_DATA]
    venue_chase_pct = {v['venue']: v['chase_pct'] / 100.0 for v in VENUE_DATA}
    venue_avg_score = {v['venue']: v['avg_1st'] for v in VENUE_DATA}
    venue_home = {v['venue']: v['home'] for v in VENUE_DATA}
    
    records = []
    match_id = 1
    
    for season in range(2008, 2026):
        teams = teams_by_season.get(season, teams_by_season[2025])
        n_teams = len(teams)
        matches_per_season = 60 if n_teams <= 8 else 74
        
        for m in range(matches_per_season):
            t1, t2 = np.random.choice(teams, 2, replace=False)
            venue = np.random.choice(venues)
            
            # Toss
            toss_winner = np.random.choice([t1, t2])
            chase_pct = venue_chase_pct.get(venue, 0.52)
            toss_decision = 'field' if np.random.random() < chase_pct else 'bat'
            
            # Determine winner based on team strengths + venue + toss
            s1 = team_strength.get(t1, 0.48)
            s2 = team_strength.get(t2, 0.48)
            
            # Home advantage
            home_team = venue_home.get(venue, '')
            if t1 == home_team:
                s1 += 0.08
            elif t2 == home_team:
                s2 += 0.08
            
            # Toss advantage
            if toss_winner == t1:
                s1 += 0.03
            else:
                s2 += 0.03
            
            # Chase advantage at venue
            batting_first = toss_winner if toss_decision == 'bat' else (t2 if toss_winner == t1 else t1)
            chasing_team = t2 if batting_first == t1 else t1
            
            p1 = s1 / (s1 + s2)
            
            # Add some noise for realism
            p1 += np.random.normal(0, 0.05)
            p1 = np.clip(p1, 0.15, 0.85)
            
            winner = t1 if np.random.random() < p1 else t2
            loser = t2 if winner == t1 else t1
            
            avg_score = venue_avg_score.get(venue, 175)
            first_score = int(np.random.normal(avg_score, 20))
            first_score = max(90, min(260, first_score))
            first_wickets = np.random.randint(3, 11)
            
            if winner == batting_first:
                second_score = first_score - np.random.randint(5, 50)
                second_wickets = 10
                margin_runs = first_score - second_score
                margin_wickets = 0
            else:
                second_score = first_score + 1
                second_wickets = np.random.randint(2, 9)
                margin_runs = 0
                margin_wickets = 10 - second_wickets
            
            dew = 1 if np.random.random() < 0.3 else 0
            
            records.append({
                'match_id': match_id,
                'season': season,
                'match_number': m + 1,
                'date': f"{season}-{np.random.randint(3,6):02d}-{np.random.randint(1,29):02d}",
                'team1': t1,
                'team2': t2,
                'venue': venue,
                'city': next((v['city'] for v in VENUE_DATA if v['venue'] == venue), 'Unknown'),
                'toss_winner': toss_winner,
                'toss_decision': toss_decision,
                'first_innings_score': first_score,
                'first_innings_wickets': first_wickets,
                'second_innings_score': second_score,
                'second_innings_wickets': second_wickets,
                'winner': winner,
                'win_margin_runs': margin_runs,
                'win_margin_wickets': margin_wickets,
                'player_of_match': '',
                'dew_factor': dew,
            })
            match_id += 1
    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(PROCESSED_DIR, 'ipl_matches_2008_2025.csv'), index=False)
    print(f"Generated {len(df)} historical matches (2008-2025)")
    return df


def parse_cricsheet_csv(data_dir):
    """Parse Cricsheet CSV2 format into match-level dataframe."""
    info_files = []
    csv_dir = os.path.join(data_dir, 'ipl_csv2')
    
    if not os.path.exists(csv_dir):
        return None
    
    all_matches = []
    for fname in sorted(os.listdir(csv_dir)):
        if fname.endswith('_info.csv') or not fname.endswith('.csv'):
            continue
        fpath = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if len(df) > 0:
                match_info = extract_match_info(df, fname)
                if match_info:
                    all_matches.append(match_info)
        except Exception:
            continue
    
    if all_matches:
        result = pd.DataFrame(all_matches)
        result.to_csv(os.path.join(PROCESSED_DIR, 'ipl_matches_2008_2025.csv'), index=False)
        print(f"Parsed {len(result)} matches from Cricsheet")
        return result
    return None


def extract_match_info(df, fname):
    """Extract match-level info from a Cricsheet CSV."""
    try:
        info = {}
        info['match_id'] = fname.replace('.csv', '')
        
        if 'info' in df.columns:
            info_rows = df[df.iloc[:, 0] == 'info']
            for _, row in info_rows.iterrows():
                key = str(row.iloc[1]).strip() if len(row) > 1 else ''
                val = str(row.iloc[2]).strip() if len(row) > 2 else ''
                if key == 'season':
                    info['season'] = int(val) if val.isdigit() else 2020
                elif key == 'team':
                    if 'team1' not in info:
                        info['team1'] = val
                    else:
                        info['team2'] = val
                elif key == 'toss_winner':
                    info['toss_winner'] = val
                elif key == 'toss_decision':
                    info['toss_decision'] = val
                elif key == 'venue':
                    info['venue'] = val
                elif key == 'city':
                    info['city'] = val
                elif key == 'winner':
                    info['winner'] = val
                elif key == 'date':
                    info['date'] = val
        return info if 'winner' in info else None
    except Exception:
        return None


def save_2026_data():
    """Save IPL 2026 current season data."""
    pd.DataFrame(POINTS_TABLE_2026).to_csv(
        os.path.join(PROCESSED_DIR, 'ipl_2026_points_table.csv'), index=False)
    pd.DataFrame(BATTING_LEADERS_2026).to_csv(
        os.path.join(PROCESSED_DIR, 'ipl_2026_batting_leaders.csv'), index=False)
    pd.DataFrame(BOWLING_LEADERS_2026).to_csv(
        os.path.join(PROCESSED_DIR, 'ipl_2026_bowling_leaders.csv'), index=False)
    pd.DataFrame(VENUE_DATA).to_csv(
        os.path.join(PROCESSED_DIR, 'ipl_venues.csv'), index=False)
    print("Saved IPL 2026 season data.")


if __name__ == '__main__':
    print("=" * 60)
    print("IPL ML Prediction Engine — Data Pipeline")
    print("=" * 60)
    
    # Try downloading real data first
    success = download_cricsheet_data()
    
    if success:
        df = parse_cricsheet_csv(RAW_DIR)
        if df is None:
            print("Cricsheet parse failed, generating synthetic data...")
            df = generate_historical_data()
    else:
        df = generate_historical_data()
    
    save_2026_data()
    
    print(f"\nData pipeline complete!")
    print(f"  Matches: {len(df)}")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Files saved to: {PROCESSED_DIR}")
