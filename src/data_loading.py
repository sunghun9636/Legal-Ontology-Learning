from os.path import join
import pandas as pd
from preprocess import normalize, text_to_tokens, replace_ner

# load json file data into dataframe
def data_load(data_path):
    with open(data_path) as data_file:
        df = pd.read_json(data_file, lines=True) # read each line of the file as JSON item

    # print(df.head(5))
    # print(df.shape)
    # print(df.describe())
    # print(df["casebody"][0]["data"])

    return df


def load_and_preprocess(data_path):
    # loading the data from json into df
    data = data_load(data_path)

    # preprocessing the text
    for i in range(len(data["casebody"])):
        data["casebody"][i]["data"] = normalize(text_to_tokens(data["casebody"][i]["data"]))

    return data


def main():
    path = join("data", "Arkansas-20181204-xml", "data", "data.json") # path to the data

    # print(load_and_preprocess(path)["casebody"][10]["data"])

    df = data_load(path)
    replace_ner(df["casebody"][100]["data"])


if __name__ == '__main__':
    main()
