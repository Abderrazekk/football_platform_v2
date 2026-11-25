# Data Dictionary — Football Intelligence Platform (v3.0)

| File | Column | Meaning | Type | Used For |
| :--- | :--- | :--- | :--- | :--- |
| `E0_Season_XXXXYYYY.csv` | `Match_Date` | Match date (`DD/MM/YYYY`) | string/date | Chronological sorting, rest days, split |
| `E0_Season_XXXXYYYY.csv` | `Home_Team` / `Away_Team` | Home and away club names | string | Team mapping, form, H2H, clustering |
| `E0_Season_XXXXYYYY.csv` | `Full_Time_Home_Goals` / `Full_Time_Away_Goals` | Goals scored at full time | int | Past form features, goal difference, clustering |
| `E0_Season_XXXXYYYY.csv` | `Full_Time_Result` | Match outcome (`H`, `D`, `A`) | string | **Target variable** for Match Classification |
| `E0_Season_XXXXYYYY.csv` | `Home_Shots`, `Home_Shots_On_Target` (and Away) | Total shots and shots on target | int | Attacking/defensive team features & clustering |
| `E0_Season_XXXXYYYY.csv` | `Home_Corners`, `Home_Yellow_Cards`, `Home_Red_Cards` (and Away) | Set pieces and discipline | int | Team profile clustering |
| `master_gameweeks.csv` | `name`, `element`, `position`, `team` | Player identity, FPL ID, role (`GK/DEF/MID/FWD`), club | string/int | Player search, profiles, one-hot encoding |
| `master_gameweeks.csv` | `round`, `kickoff_time`, `was_home`, `opponent_team` | Gameweek number, date, venue, opponent ID | int/bool/date | Chronological player split & fixture context |
| `master_gameweeks.csv` | `minutes`, `goals_scored`, `assists`, `bps`, `ict_index`, `xP` | Per-gameweek player performance metrics | int/float | Rolling 3 & 5 GW features for Player Regression |
| `master_gameweeks.csv` | `total_points` | FPL points earned in that gameweek | int | **Target variable** for Player Regression |
| `master_gameweeks.csv` | `value` | Player price (in tenths of £M, e.g., `55` = £5.5M) | int | Greedy Best XI 100M budget constraint |
| `master_player_gw.csv` | `player_folder_name` | Unique folder string (`Name_ID`) | string | Cross-checking player identity |
| `master_player_history.csv` | `element_code` | Persistent player ID across seasons | int | Linking player history across club transfers |
| `master_player_history.csv` | `season_name`, `minutes`, `goals_scored`, `assists`, `total_points` | Past seasonal totals per player | string/int | Features & **Target** for Season Ridge/Lasso |
| `master_understat_teams.csv` | `team_name`, `xG`, `xGA`, `ppda`, `date` | Expected goals & pressing intensity (2020-25 only) | string/float | Optional exploration (excluded from 2025-27 models) |
| `master_xP_projections.csv` | `id`, `xP` | Player element ID and expected points | int/float | Complementary player projection feature |