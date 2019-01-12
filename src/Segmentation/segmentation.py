class Segmantor:
    def __init__(self, raw_pointcloud):
        self.raw_pointcloud = raw_pointcloud
        self.__segment()

    def __segment(self):
        #do the magic
        self.labled_pointcloud = self.raw_pointcloud
    