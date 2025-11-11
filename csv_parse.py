

import pandas as pd

class CSVParser:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.triggers = sorted(self.df["Trigger"].unique())
        self.channels = [col for col in self.df.columns if col.startswith("EEG")]

    def give_triggers(self):
        return self.triggers

    def get_channel(self, trigger_value, channel_name):
        if channel_name not in self.channels:
            raise ValueError(f"Ungültiger Kanalname: {channel_name}")
        subset = self.df[self.df["Trigger"] == trigger_value]
        return subset[channel_name].to_numpy()

# Beispiel:
# parser = CSVParser("data-parsed/Data_20251031_105214_Samuel.csv")
# print(parser.give_triggers())
# x = parser.get_channel(1, "EEG 2")