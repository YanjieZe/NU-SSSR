import cv2
import utils
from sampling import DelaunayTriangulation, DelaunayTriangulationBlur

if __name__=='__main__':
    img_path = 'imgs/img_origin.jpeg'
    img = cv2.imread(img_path)
    img_delaunay = DelaunayTriangulation(img, 2000)
    
    utils.show_img(img_delaunay)

    img_delaunay = DelaunayTriangulationBlur(img,10000, method="center")
    utils.show_img(img_delaunay)