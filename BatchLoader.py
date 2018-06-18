from multiprocessing.dummy import Process as Thread, Lock, Pool as ThreadPool
from queue import Queue
import random


class _BatchLoaderIter(object):
    def __init__(self, loader):
        self.batch_size = loader.batch_size
        self.dataset = loader.dataset
        self.batch_buffer = Queue(loader.pre_fetch)

        self.fetcher = Thread(target=self._batch_loader_fetcher)
        self.fetcher.start()
        self.all_data_fetched = False

    def _batch_loader_fetcher(self):
        idxs = list(range(len(self.dataset)))
        random.shuffle(idxs)
        steps_per_epoch = int(len(self.dataset) / self.batch_size)
        pool = ThreadPool(4)

        for step in range(steps_per_epoch):
            start_idx = step * self.batch_size
            end_idx = (step + 1) * self.batch_size
            images = pool.map(lambda idx: self.dataset[idxs[idx]], range(start_idx, end_idx))
            self.batch_buffer.put(images)

        self.all_data_fetched = True

    def __iter__(self):
        return self

    def __next__(self):
        batch = None
        while not self.all_data_fetched:
            try:
                batch = self.batch_buffer.get_nowait()
                break
            except Exception:
                pass

        if batch is None:
            raise StopIteration

        return batch


class BatchLoader(object):
    def __init__(self, dataset, batch_size=32, pre_fetch=6):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pre_fetch = pre_fetch
        pass

    def __iter__(self):
        return _BatchLoaderIter(self)
