from multiprocessing import Process, Manager
from ctypes import c_bool
import random


def _batch_loader_feeder(size, q, stop_flag):
    idxs = list(range(size))
    random.shuffle(idxs)
    for idx in idxs:
        q.put(idx)

    stop_flag.value = True


def _batch_loader_fetcher(dataset, input_queue, buffer_queue, stop_flag, finished_fetchers, finished_fetcher_lock):
    while True:
        if stop_flag.value is True and input_queue.empty():
            break

        try:
            idx = input_queue.get_nowait()
            buffer_queue.put(dataset[idx])
        except Exception:
            pass

    finished_fetcher_lock.acquire()
    finished_fetchers.value += 1
    finished_fetcher_lock.release()


class _BatchLoaderIter(object):
    def __init__(self, loader):
        self.batch_size = loader.batch_size
        queue_size = loader.batch_size * loader.pre_fetch
        self.manager = Manager()
        self.input_queue = self.manager.Queue(maxsize=queue_size)
        self.output_queue = self.manager.Queue(maxsize=queue_size)
        self.stop_flag = self.manager.Value(c_bool, False)

        self.feeder = Process(target=_batch_loader_feeder,
                              args=(len(loader.dataset), self.input_queue, self.stop_flag))

        self.finished_fetchers = self.manager.Value('i', 0)
        self.finished_fetcher_lock = self.manager.Lock()
        self.fetchers = [
            Process(target=_batch_loader_fetcher,
                    args=(loader.dataset, self.input_queue, self.output_queue, self.stop_flag,
                          self.finished_fetchers, self.finished_fetcher_lock))
            for _ in range(loader.num_workers)
        ]

        self.feeder.start()
        for fetcher in self.fetchers:
            fetcher.start()

    def __iter__(self):
        return self

    def __next__(self):
        buffer = []
        while len(buffer) < self.batch_size and \
                not (self.finished_fetchers.value == len(self.fetchers) and
                     self.output_queue.empty()):
            try:
                e = self.output_queue.get_nowait()
                buffer.append(e)
            except Exception:
                pass

        if len(buffer) != self.batch_size:
            raise StopIteration

        return buffer


class BatchLoader(object):
    def __init__(self, dataset, batch_size=32, num_workers=2, pre_fetch=6):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_fetch = pre_fetch
        pass

    def __iter__(self):
        return _BatchLoaderIter(self)
