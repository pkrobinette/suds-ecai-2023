from utils.utils import load_ddh_mnist, load_udh_mnist

if __name__ == "__main__":
    HnetD, RnetD = load_ddh_mnist()
    Hnet, Rnet = load_udh_mnist()