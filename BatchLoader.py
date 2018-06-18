from multiprocessing.dummy import Process as Thread, Lock, Pool as ThreadPool
from queue import Queue
import random


class _BatchLoaderIter(object):
    def __init__(self, loader):
        self.batch_size = loader.batch_size
        self.num_threads = loader.num_threads
        self.dataset = loader.dataset
        self.op_fn = loader.op_fn
        self.batch_buffer = Queue(loader.pre_fetch)
        self.steps = int(len(self.dataset) / self.batch_size)

        self.fetcher = Thread(target=self._batch_loader_fetcher)
        self.fetcher.start()
        self.all_data_fetched = False
        self.mutex = Lock()

    def _batch_loader_fetcher(self):
        idxs = list(range(len(self.dataset)))
        random.shuffle(idxs)
        pool = ThreadPool(self.num_threads)

        for step in range(self.steps):
            start_idx = step * self.batch_size
            end_idx = (step + 1) * self.batch_size
            images = pool.map(lambda idx: self.dataset[idxs[idx]], range(start_idx, end_idx))
            images = self.op_fn(images) if self.op_fn is not None else images
            self.batch_buffer.put(images)

        self.mutex.acquire()
        self.all_data_fetched = True
        self.mutex.release()

    def __iter__(self):
        return self

    def __next__(self):
        self.mutex.acquire()

        if self.all_data_fetched and self.batch_buffer.empty():
            self.mutex.release()
            raise StopIteration

        self.mutex.release()
        return self.batch_buffer.get()

    def __del__(self):
        self.fetcher.join()


class BatchLoader(object):
    def __init__(self, dataset, batch_size=32, pre_fetch=6, num_threads=4, op_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pre_fetch = pre_fetch
        self.num_threads = num_threads
        self.op_fn = op_fn

    def __iter__(self):
        return _BatchLoaderIter(self)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)
