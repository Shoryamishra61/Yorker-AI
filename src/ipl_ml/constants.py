from __future__ import annotations

CURRENT_SEASON = 2026
TRAIN_END_SEASON = 2024
VALIDATION_SEASON = 2025

CRICSHEET_IPL_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"
IPL_COMPETITION_URL = "https://scores.iplt20.com/ipl/mc/competition.js"

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

VENUE_ALIASES = {
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "ACA Stadium": "Barsapara Cricket Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Himachal Pradesh Cricket Association Stadium": "HPCA Cricket Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "HPCA Cricket Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M. Chinnaswamy Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "New International Cricket Stadium": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Cricket Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Cricket Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Cricket Stadium",
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "M.A. Chidambaram Stadium",
    "MA Chidambaram Stadium": "M.A. Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "M.A. Chidambaram Stadium",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Ekana Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Ekana Cricket Stadium",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "Rajiv Gandhi Intl. Cricket Stadium": "Rajiv Gandhi International Cricket Stadium",
    "Maharaja YS International Cricket Stadium": "Maharaja Yadavindra Singh International Cricket Stadium",
}

TEAM_HOME_VENUES = {
    "Chennai Super Kings": {"M.A. Chidambaram Stadium"},
    "Delhi Capitals": {"Arun Jaitley Stadium"},
    "Gujarat Titans": {"Narendra Modi Stadium"},
    "Kolkata Knight Riders": {"Eden Gardens"},
    "Lucknow Super Giants": {"Ekana Cricket Stadium"},
    "Mumbai Indians": {"Wankhede Stadium"},
    "Punjab Kings": {"Maharaja Yadavindra Singh International Cricket Stadium", "HPCA Cricket Stadium"},
    "Rajasthan Royals": {"Sawai Mansingh Stadium", "Barsapara Cricket Stadium", "ACA Cricket Stadium"},
    "Royal Challengers Bengaluru": {"M. Chinnaswamy Stadium"},
    "Sunrisers Hyderabad": {"Rajiv Gandhi International Cricket Stadium"},
}

VENUE_METADATA = {
    "M. Chinnaswamy Stadium": {
        "city": "Bengaluru",
        "pitch_type": "batting-friendly",
        "avg_first_innings_score": 185.0,
        "chase_win_pct": 0.52,
        "spin_bias": 0.38,
        "pace_bias": 0.62,
    },
    "M.A. Chidambaram Stadium": {
        "city": "Chennai",
        "pitch_type": "spin-friendly",
        "avg_first_innings_score": 161.0,
        "chase_win_pct": 0.44,
        "spin_bias": 0.70,
        "pace_bias": 0.30,
    },
    "Wankhede Stadium": {
        "city": "Mumbai",
        "pitch_type": "balanced",
        "avg_first_innings_score": 178.0,
        "chase_win_pct": 0.54,
        "spin_bias": 0.42,
        "pace_bias": 0.58,
    },
    "Eden Gardens": {
        "city": "Kolkata",
        "pitch_type": "batting-friendly",
        "avg_first_innings_score": 182.0,
        "chase_win_pct": 0.55,
        "spin_bias": 0.40,
        "pace_bias": 0.60,
    },
    "Narendra Modi Stadium": {
        "city": "Ahmedabad",
        "pitch_type": "balanced",
        "avg_first_innings_score": 175.0,
        "chase_win_pct": 0.51,
        "spin_bias": 0.48,
        "pace_bias": 0.52,
    },
    "Arun Jaitley Stadium": {
        "city": "Delhi",
        "pitch_type": "batting-friendly",
        "avg_first_innings_score": 188.0,
        "chase_win_pct": 0.56,
        "spin_bias": 0.36,
        "pace_bias": 0.64,
    },
    "Rajiv Gandhi International Cricket Stadium": {
        "city": "Hyderabad",
        "pitch_type": "batting-friendly",
        "avg_first_innings_score": 190.0,
        "chase_win_pct": 0.58,
        "spin_bias": 0.35,
        "pace_bias": 0.65,
    },
    "HPCA Cricket Stadium": {
        "city": "Dharamsala",
        "pitch_type": "pace-friendly",
        "avg_first_innings_score": 170.0,
        "chase_win_pct": 0.50,
        "spin_bias": 0.28,
        "pace_bias": 0.72,
    },
    "Maharaja Yadavindra Singh International Cricket Stadium": {
        "city": "New Chandigarh",
        "pitch_type": "balanced",
        "avg_first_innings_score": 176.0,
        "chase_win_pct": 0.53,
        "spin_bias": 0.42,
        "pace_bias": 0.58,
    },
    "Ekana Cricket Stadium": {
        "city": "Lucknow",
        "pitch_type": "balanced",
        "avg_first_innings_score": 172.0,
        "chase_win_pct": 0.52,
        "spin_bias": 0.48,
        "pace_bias": 0.52,
    },
    "ACA Cricket Stadium": {
        "city": "Guwahati",
        "pitch_type": "balanced",
        "avg_first_innings_score": 168.0,
        "chase_win_pct": 0.50,
        "spin_bias": 0.45,
        "pace_bias": 0.55,
    },
    "Barsapara Cricket Stadium": {
        "city": "Guwahati",
        "pitch_type": "balanced",
        "avg_first_innings_score": 165.0,
        "chase_win_pct": 0.50,
        "spin_bias": 0.52,
        "pace_bias": 0.48,
    },
    "Sawai Mansingh Stadium": {
        "city": "Jaipur",
        "pitch_type": "balanced",
        "avg_first_innings_score": 174.0,
        "chase_win_pct": 0.51,
        "spin_bias": 0.44,
        "pace_bias": 0.56,
    },
}

EXPLAINER_TOP_N = 5
SIMULATION_RUNS = 2000
