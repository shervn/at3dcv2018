from Augmentation.augmentation import Augmentor
from Segmentation.segmentation import Segmantor
from Reconstruction.reconstruction import Reconstructor

if __name__ == "__main__":

    r = Reconstructor()
    s = Segmantor(r.reconstructed_pointcloud)
    t = Augmentor(s.labled_pointcloud)
    