from os.path import join
import pandas as pd
from preprocess import normalize, text_to_tokens


def data_load(data_path): # load json file data into dataframe
    with open(data_path) as data_file:
        df = pd.read_json(data_file, lines=True) # read each line of the file as JSON item

    # print(df["casebody"][0]["data"])

    return df


def load_and_preprocess(data_path):
    # loading the data from json into df
    data = data_load(data_path)

    # preprocessing the text
    for i in range(len(data["casebody"])):
        data["casebody"][i]["data"] = normalize(text_to_tokens(data["casebody"][i]["data"]))
        # break

    return data


def main():
    path = join("data", "Arkansas-20181204-xml", "data", "data.json") # path to the data

    print(load_and_preprocess(path)["casebody"][0]["data"])


if __name__ == '__main__':
    main()
