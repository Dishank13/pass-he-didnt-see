"""Download StatsBomb open data (events + 360) for all matches with 360 coverage."""
from phds.data.load_statsbomb import download_all

if __name__ == "__main__":
    m = download_all()
    print(f"{len(m)} matches with 360 data")
    print(m.groupby(["competition_competition_name", "season_season_name"]).size().to_string())
