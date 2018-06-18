from multiprocessing.dummy import Process as Thread, Lock
from queue import Queue
import random


class _BatchLoaderIter(object):
    def __init__(self, loader):
        self.batch_size = loader.batch_size
        self.dataset = loader.dataset
        queue_size = loader.batch_size * loader.pre_fetch
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size)
        self.stop_flag = False

        self.feeder = Thread(target=self._batch_loader_feeder)

        self.finished_fetchers = 0
        self.finished_fetcher_lock = Lock()
        self.fetchers = [
            Thread(target=self._batch_loader_fetcher)
            for _ in range(loader.num_workers)
        ]
        self.batch_exhausted = False
        self.batch_queue = Queue(maxsize=loader.pre_fetch)
        self.collector = Thread(target=self._batch_loader_collector)

        self.feeder.start()
        self.collector.start()
        for fetcher in self.fetchers:
            fetcher.start()

    def _batch_loader_feeder(self):
        idxs = list(range(len(self.dataset)))
        random.shuffle(idxs)
        for idx in idxs:
            self.input_queue.put(idx)

        self.stop_flag = True

    def _batch_loader_fetcher(self):
        while True:
            if self.stop_flag is True and self.input_queue.empty():
                break

            try:
                idx = self.input_queue.get_nowait()
                self.output_queue.put(self.dataset[idx])
            except Exception:
                pass

        self.finished_fetcher_lock.acquire()
        self.finished_fetchers += 1
        self.finished_fetcher_lock.release()

    def _batch_loader_collector(self):
        while True:
            buffer = []
            while len(buffer) < self.batch_size and \
                    not (self.finished_fetchers == len(self.fetchers) and
                         self.output_queue.empty()):
                try:
                    e = self.output_queue.get_nowait()
                    buffer.append(e)
                except Exception:
                    pass
            if len(buffer) != self.batch_size:
                break
            self.batch_queue.put(buffer)

        self.batch_exhausted = True

    def __iter__(self):
        return self

    def __next__(self):
        batch = None
        while not self.batch_exhausted:
            try:
                batch = self.batch_queue.get_nowait()
                break
            except Exception:
                pass

        if batch is None:
            raise StopIteration

        return batch


class BatchLoader(object):
    def __init__(self, dataset, batch_size=32, num_workers=2, pre_fetch=6):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_fetch = pre_fetch
        pass

    def __iter__(self):
        return _BatchLoaderIter(self)
