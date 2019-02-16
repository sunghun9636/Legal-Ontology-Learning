from os.path import join
import pandas as pd
from preprocess import tokenize, normalize

# load json file data into dataframe
def data_load(data_path):
    with open(data_path) as data_file:
        df = pd.read_json(data_file, lines=True) # read each line of the file as JSON item

    # print(df.head(5))
    # print(df.shape)
    # print(df.describe())
    # print(df["casebody"][0]["data"])

    return df

def main():
    path = join("data", "Arkansas-20181204-xml", "data", "data.json") # path to the data

    df = data_load(path)

    words = tokenize(df["casebody"][0]["data"])
    print(normalize(words))


if __name__ == '__main__':
    main()
