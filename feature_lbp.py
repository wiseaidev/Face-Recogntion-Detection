import numpy as np
from matplotlib import pyplot as plt
import cv2 
from LBP import Fast_LBP
class features_extraction():
    def __init__(self, lbp, grid_size = (8,8)):
        self._lbp = lbp
        self._grid_size = grid_size
    def lbp_histograms(self, image):
        # compute the lbp of the image
        LBP = self._lbp.Compute_LBP(image)
        print(LBP)
        # divide the lbp imgae into small regions
        LBP_height, LBP_width = LBP.shape
        Grid_rows, Grid_cols = self._grid_size
        row_number = int(LBP_height/Grid_rows)
        col_number = int(LBP_width/Grid_cols)
        features = []
        for row in range(0,Grid_rows):
            cv2.line(image, (int(row*row_number),0), (int(row*row_number),255), (0, 0, 0) , 2) 
            cv2.line(image, (0,int(row*row_number)), (255,int(row*row_number)), (0, 0, 0) , 2)
            for col in range(0,Grid_cols):
                Chunk_LBP = LBP[row*row_number:(row+1)*row_number,col*col_number:(col+1)*col_number]
                #compute the histogram for each region
                (Histogram, bins) = np.histogram(Chunk_LBP, bins=2**8,
                 range=(0, 2**8))
                width = bins[1] - bins[0]
                center = (bins[:-1] + bins[1:]) / 2
                axs[row][col].bar(center, Histogram, align='center', width=width)
                np.append(features, Histogram, 0)
                print(row)
        return np.asarray(features)
if __name__ == "__main__":
    img = np.array(cv2.imread("images/img3.jpg",cv2.IMREAD_GRAYSCALE), 'uint8')
    print(img)
    fig, axs = plt.subplots(8,8, figsize=(100,100), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    lbp = Fast_LBP(1,8)
    features_extraction = features_extraction(lbp)
    features_extraction.lbp_histograms(img)

    plt.show()
    cv2.imshow('image', img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        pass    


