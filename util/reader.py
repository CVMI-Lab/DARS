
class DataReader(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)


    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            self.construct_iter()
            return self.dataloader_iter.next()

class DataReader_orig(object):
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler

    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)

    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            if self.sampler is not None:
                self.sampler.set_epoch(self.cur_epoch)
            self.construct_iter()
            return self.dataloader_iter.next()