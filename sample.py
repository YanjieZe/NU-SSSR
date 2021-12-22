import cv2
import utils
from sampling import DelaunayTriangulation, DelaunayTriangulationBlur

if __name__=='__main__':
    img_path = 'imgs/img_origin.jpeg'
    img = cv2.imread(img_path)
    # img_delaunay = DelaunayTriangulation(img, 2000)
    
    # utils.show_img(img_delaunay)
    
    img_delaunay = DelaunayTriangulationBlur(img, 1000, method="center", sample_point_method="fourier")
    utils.show_img(img_delaunay)
    # utils.save_img('imgs/random_triangulation.png', img_delaunay)
    utils.save_img('imgs/fourier_triangulation.png', img_delaunay)
