import cv2
from sampling import DelaunayTriangulation

if __name__=='__main__':
    img_path = 'imgs/img_test.jpeg'
    img = cv2.imread(img_path)
    img_delaunay, img_voronoi = DelaunayTriangulation(img)
    
    cv2.imwrite('imgs/delaunay.jpg', img_delaunay)
    cv2.imwrite('imgs/voronoi.jpg', img_voronoi)