import requests
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os

BASE_URL = "https://api.collegefootballdata.com"
# API_KEY = "sz55AwUuro/KMnA8lmg3rzeoEa7LsKLJDZQpDKNpkDqhJF26ErBICm4GJGEyMJku"
load_dotenv()
API_KEY = os.getenv("API_KEY")

## IN PROGRESS ##
def get_fbs_records_by_team_by_year(year, team):
    api_key = API_KEY
    url = f"{BASE_URL}/teams/records?year={year}&team={team}"
    headers = {"Authorization": f"Bearer {api_key}"}
## IN PROGRESS ##

def get_fbs_teams_by_year(year):

    api_key = API_KEY
    url = f"{BASE_URL}/teams/fbs?year={year}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return pd.DataFrame(response.json())

def get_games_by_year_week(year, week=None, season_type='regular'):

    api_key = API_KEY
    url = f"{BASE_URL}/games"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "year": year,
        "seasonType": season_type,
    }

    if week:
        params["week"] = week
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return pd.DataFrame(response.json())

def process_game_data(games_df, fbs_teams_df):

    teams = fbs_teams_df['school'].tolist()
    games = []
    fcs_losses = []

    game_counter = {}

    records = {}
    for team in teams:
        records[team] = [0, 0]

    for _, row in games_df.iterrows():
        team1, team2 = row['homeTeam'], row['awayTeam']
        team1_score, team2_score = row['homePoints'], row['awayPoints']
        if pd.isnull(team1_score) or pd.isnull(team2_score):
            continue  # Skip games without scores

        i, j = team1, team2
        key = tuple(sorted([i, j]))  # unordered pair
        game_counter[key] = game_counter.get(key, 0)
        k = game_counter[key]
        game_counter[key] += 1

        team1_is_fbs = team1 in teams
        team2_is_fbs = team2 in teams

        margin = abs(team1_score - team2_score)
        winner = team1 if team1_score > team2_score else team2
        location = "neutral" if row.get("neutralSite", False) else "home"
        alpha = 1 if location == "neutral" else (0.8 if row['homeTeam'] == winner else 1.2)

        # Track FCS losses
        if team1_is_fbs and not team2_is_fbs and team1_score < team2_score:
            fcs_losses.append((team1, margin, alpha, row.get("week"), row.get("season")))
        elif team2_is_fbs and not team1_is_fbs and team2_score < team1_score:
            fcs_losses.append((team2, margin, alpha, row.get("week"), row.get("season")))
        elif team1_is_fbs and team2_is_fbs:
            if team1 == winner:
                records[team1][0] += 1
                records[team2][1] += 1
            elif team2 == winner:
                records[team1][1] += 1
                records[team2][0] += 1
            games.append((
                team1,
                team2,
                k,
                winner,
                margin,
                alpha,
                row.get("week"),
                row.get("season")
            ))

    return teams, games, fcs_losses, records