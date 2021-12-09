#!/usr/bin/python

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import core
import numba as nb

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0 )

# 从一个三角形区域采集色彩
def sample_mean_color(img, pt1, pt2, pt3, method="center"):
    points = np.array([pt1, pt2, pt3])
    rect = cv2.minAreaRect(points)
    ((cx, cy), (width, height), angle) = rect
    cx = int(cx)
    cy = int(cy)
    color = None

    # center，直接用矩形中心点作为近似的颜色
    if method=="center":
        color = img[cy,cx,:]
        color = (int(color[0]), int(color[1]), int(color[2]))
    elif method=="vertex":
        color = img[pt1[1], pt1[0],:]
        color += img[pt2[1], pt2[0],:]
        color += img[pt3[1], pt3[0],:]
        color = (int(color[0]/3), int(color[1]/3), int(color[2]/3))

    elif method=="random":
        raise NotImplemented('Method random not implemented.')
    else:
        raise Exception('Method not supported.')
    return color

# Draw delaunay triangles
def draw_delaunay_blur(img, subdiv, method) :

    triangleList = subdiv.getTriangleList() 
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            color = sample_mean_color(img, pt1, pt2, pt3, method)

            triangle_cnt = np.array([pt1, pt2, pt3])
            # 通过轮廓填充来填充三角形
            cv2.drawContours(img, [triangle_cnt], 0, tuple(color), -1)

# Draw voronoi diagram
def draw_voronoi(img, subdiv) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0) 
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)

        cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)

@nb.jit()
def DelaunayTriangulationBlur(img, point_num=1000, method="center"):
    """
    对img进行三角采样，并使用三角中的颜色进行填充，支持的颜色方法：
    center，取三角中心点;
    random，随机在三角形采样;
    vertex，取三角形点进行平均;
    """
    # if img is None:
    #     raise Exception('Img can not be None.')
    # if not isinstance(img, np.ndarray):
    #     raise Exception('Input should be img/array.')
    # Turn on animation while drawing triangles

    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)


    # Keep a copy around
    img_orig = img.copy() 

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect) 
    
    # Create an array of points.
    # points = set()
    points = []
    
    # # generate points randomly
    width = img.shape[0]
    height = img.shape[1]
    for i in range(point_num):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        points.append((y,x))
    # sampleOnePoint = lambda width, height : (np.random.randint(0, height), np.random.randint(0, width))
    # samplePoints = np.vectorize(sampleOnePoint)
    # points = samplePoints([width]*point_num, [width]*point_num)
    # x = np.random.randn(width)
    # y = np.random.randn(height)
    # print(x.shape, y.shape)
    # points = np.meshgrid(x, y)
    # print(points[0].shape, points[1].shape)
    # result = list(map(tuple, points))
    # print(len(result))

    
    # # Insert points into subdiv
    for p in points :
        # import pdb; pdb.set_trace()
        subdiv.insert(p)
    # print(len(points), points)
    # subdiv.insert(points)

    # Draw delaunay triangles
    draw_delaunay_blur( img, subdiv, method ) 

    is_draw_points=False
    if is_draw_points:
        # Draw points
        for p in points :
            draw_point(img, p, (0,0,255))

    return img



if __name__ == '__main__':

    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    # Turn on animation while drawing triangles
    animate = False

    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)

    # Read in the image.
    img = cv2.imread("../imgs/img_origin.jpeg") 

    # Keep a copy around
    img_orig = img.copy() 

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect) 
    
    # Create an array of points.
    points = set()
    

    # 随机采点
    width = img.shape[0]
    height = img.shape[1]
    for i in range(1000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        points.add((y,x))

   

    # Insert points into subdiv
    for p in points :
        # import pdb; pdb.set_trace()
        subdiv.insert(p)

        # Show animation
        if animate :
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay_blur( img_copy, subdiv, "center") 
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    # Draw delaunay triangles
    draw_delaunay_blur( img, subdiv, "center" ) 

    is_draw_points=False
    if is_draw_points:
        # Draw points
        for p in points :
            draw_point(img, p, (0,0,255))

    # Allocate space for Voronoi Diagram
    img_voronoi = np.zeros(img.shape, dtype = img.dtype)

    # Draw Voronoi diagram
    draw_voronoi(img_voronoi,subdiv)

    # Show results
    # cv2.imshow(win_delaunay,img)
    cv2.imwrite('../imgs/delaunay_blur.jpg', img)
    # cv2.imshow(win_voronoi,img_voronoi)
    # cv2.imwrite('imgs/voronoi.jpg', img)