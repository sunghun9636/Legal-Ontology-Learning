from os.path import join
import pandas as pd
from preprocess import normalize, text_to_tokens
import pickle


def data_load(data_path):  # load json file data into dataframe
    with open(data_path) as data_file:
        df = pd.read_json(data_file, lines=True)  # read each line of the file as JSON item

    return df


def load_and_preprocess(data_path):
    # loading the data from json into df
    data = data_load(data_path)
    output = []

    # preprocessing the text
    # for i in range(len(data["casebody"])):
    for i in range(2500):
        print('{} {}'.format('preprocessing data number: ', i))
        data["casebody"][i]["data"] = normalize(text_to_tokens(data["casebody"][i]["data"]))
        output.append(data["casebody"][i]["data"])

    with open('data/case_documents_2500.data', 'wb') as file:
        print("... Saving the pre-processed data into local binary file...")
        # store the data as binary data stream
        pickle.dump(output, file)

    return output


def main():
    path = join("data", "Arkansas-20181204-xml", "data", "data.json")  # path to the data

    print(load_and_preprocess(path)[0])


if __name__ == '__main__':
    main()
