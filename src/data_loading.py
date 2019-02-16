import json
from os.path import join
import pandas as pd

def data_load(data_path):
    with open(data_path) as data_file:
        df = pd.read_json(data_file, lines=True) # read each line of the file as JSON item

    # print(df.head(5))
    # print(df.shape)
    # print(df.describe())
    print(df["casebody"][0]["data"])

def main():
    path = join("data", "Arkansas-20181204-xml", "data", "data.json")

    data_load(path)


if __name__ == '__main__':
    main()
