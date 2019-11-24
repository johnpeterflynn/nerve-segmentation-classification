import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, test_config=None):
        self.validation_split = validation_split
        self.test_config = test_config
        self.shuffle = shuffle
        self.test_sampler = None
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_val_sampler(
            self.validation_split)

        if test_config is not None:
            self.sampler, self.test_sampler = self._split_test_sampler(
                self.test_config['test_split'], self.sampler.indices)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_val_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _split_test_sampler(self, split, idx_train):
        if split == 0.0:
            return None, None

        n_samples = len(idx_train)
        if isinstance(split, int):
            assert split > 0
            assert split < n_samples, "test set size is configured to be larger than len(entire dataset) - len(validation set)."
            len_valid = split
        else:
            len_valid = int(n_samples * split)

        test_idx = idx_train[0:len_valid]
        train_idx = np.delete(idx_train, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.n_samples = len(train_idx)

        return train_sampler, test_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            init_kwargs = self.init_kwargs
            init_kwargs['batch_size'] = self.test_config['batch_size']
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
