import os


class DataLoader:
    def __init__(self, data_path):
        self.path = data_path

    def load_data(self):
        f = open(self.path)
        lines = f.readlines()
        columns = []
        for line in lines:
            line = line[1:-2]
            column = line.split(',')
            columns.append([int(x.strip()) for x in column])
        return columns

    def load_cols(self, file):
        f = open(file)
        lines = f.readlines()
        columns = []
        for line in lines:
            line = line[1:-2]
            column = line.split(',')
            columns.append([int(x.strip()) for x in column])
        return columns

    def load_dataset(self, dataset_path):
        files = os.listdir(dataset_path)
        dataset = []
        for file in files:
            if not os.path.isdir(file):
                dataset.append(self.load_cols(dataset_path+"/"+file))
        return dataset


if __name__ == '__main__':
    loader = DataLoader('columns.txt')
    columns = loader.load_data()
    print(columns)
