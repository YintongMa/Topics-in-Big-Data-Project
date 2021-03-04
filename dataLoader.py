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


if __name__ == '__main__':
    loader = DataLoader('columns.txt')
    columns = loader.load_data()
    print(columns)
