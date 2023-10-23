class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset


    @property
    def get_dataset(self):
        return self.dataset

    @property
    def num_samples(self):
        return self.dataset.num_samples

    def get_model_dataset(self, dataset, batch_size):
        return dataset.get_pretain_dataset(batch_size)


