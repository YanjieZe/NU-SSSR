#!/usr/bin/python

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import core
import utils
import tqdm

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


def barycentricInterpolate(aPoint, bPoint, cPoint, aColor, bColor, cColor, img, ):
    xLeft = min(aPoint[0], bPoint[0], cPoint[0])
    xRight = max(aPoint[0], bPoint[0], cPoint[0])
    yTop = min(aPoint[1], bPoint[1], cPoint[1])
    yBottom = max(aPoint[1], bPoint[1], cPoint[1])

    xs = np.arange(xLeft, xRight)
    ys = np.arange(yTop, yBottom)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()

    alphas = (-(xv-bPoint[0]) * (cPoint[1]-bPoint[1]) + (yv-bPoint[1]) * (cPoint[0]-bPoint[0])) \
        / (-(aPoint[0]-bPoint[0]) * (cPoint[1]-bPoint[1]) + (aPoint[1]-bPoint[1]) * (cPoint[0]-bPoint[0]))
        
    betas = (-(xv-cPoint[0]) * (aPoint[1]-cPoint[1]) + (yv-cPoint[1]) * (aPoint[0]-cPoint[0])) \
        / (-(bPoint[0]-cPoint[0]) * (aPoint[1]-cPoint[1]) + (bPoint[1]-cPoint[1]) * (aPoint[0]-cPoint[0]))
    gammas = 1 - alphas - betas
    
    masks = (alphas >= -1e-9) & (betas >= -1e-9) & (gammas >= -1e-9)
    
    # print(np.where(masks != masks_p))
    xv = xv[masks]
    yv = yv[masks]
    alphas = alphas[masks]
    betas = betas[masks]
    gammas = gammas[masks]
    
    def mul(color, proportion):
        return np.asmatrix(proportion).T @ np.asmatrix(color)
    colors = mul(aColor, alphas) + mul(bColor, betas) + mul(cColor, gammas)
    # print(colors.shape, img[xv, yv].shape)
    img[xv, yv] = colors
    
    # draw_point(img, [aPoint[1], aPoint[0]], (0, 0, 255))
    # draw_point(img, [bPoint[1], bPoint[0]], (0, 0, 255))
    # draw_point(img, [cPoint[1], cPoint[0]], (0, 0, 255))
    # cv2.line(img, [yBottom, xLeft], [yBottom, xRight], (0, 0, 255))
    # cv2.line(img, [yBottom, xRight], [yTop, xRight], (0, 0, 255))
    # cv2.line(img, [yTop, xRight], [yTop, xLeft], (0, 0, 255))
    # cv2.line(img, [yTop, xLeft], [yBottom, xLeft], (0, 0, 255))

# 从一个三角形区域采集色彩
def sample_mean_color(img, pt1, pt2, pt3, method="center", mask=None):
    points = np.array([pt1, pt2, pt3])
    rect = cv2.minAreaRect(points)
    ((cx, cy), (width, height), angle) = rect
    cx = int(cx)
    cy = int(cy)
    color = None

    # center，直接用矩形中心点作为近似的颜色
    if method=="center":
        if (mask is not None):
            mask[cy, cx] = 1
        color = img[cy,cx,:]
        color = (int(color[0]), int(color[1]), int(color[2]))
    elif method=="vertex":
        if (mask is not None):
            mask[pt1[1], pt1[0]] = 1
            mask[pt2[1], pt2[0]] = 1
            mask[pt3[1], pt3[0]] = 1
        # color = img[pt1[1], pt1[0],:]
        # color += img[pt2[1], pt2[0],:]
        # color += img[pt3[1], pt3[0],:]
        # color = int(color[0]/3), int(color[1]/3), int(color[2]/3)
        # print(type(img[pt1[1], pt1[0], 0]))
        color = (int((int(img[pt1[1], pt1[0], 0]) + int(img[pt2[1], pt2[0], 0]) + int(img[pt3[1], pt3[0], 0]))/3), 
                 int((int(img[pt1[1], pt1[0], 1]) + int(img[pt2[1], pt2[0], 1]) + int(img[pt3[1], pt3[0], 1]))/3), 
                 int((int(img[pt1[1], pt1[0], 2]) + int(img[pt2[1], pt2[0], 2]) + int(img[pt3[1], pt3[0], 2]))/3))
        # print(color)

    elif method=="random":
        raise NotImplemented('Method random not implemented.')
    else:
        raise Exception('Method not supported.')
    return color

# Draw delaunay triangles
def draw_delaunay_blur(img : np.ndarray, subdiv, method, mask=None) :
    # print(img.shape)

    triangleList = subdiv.getTriangleList() 
    size = img.shape
    r = (0, 0, size[1], size[0])
    
    oldImg = img.copy()
    img.fill(0)
    for t in triangleList :

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            if (method == 'barycentric'):
                if (mask is not None):
                    mask[pt1[1], pt1[0]] = 1
                    mask[pt2[1], pt2[0]] = 1
                    mask[pt3[1], pt3[0]] = 1
                barycentricInterpolate([pt1[1], pt1[0]], [pt2[1], pt2[0]], [pt3[1], pt3[0]], oldImg[pt1[1], pt1[0]], oldImg[pt2[1], pt2[0]], oldImg[pt3[1], pt3[0]], img)
                
            else:
                color = sample_mean_color(oldImg, pt1, pt2, pt3, method, mask)
                triangle_cnt = np.array([pt1, pt2, pt3])
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


def DelaunayTriangulationBlur(img, point_num=1000, method="center", \
                        sample_point_method="random", 
                        frequency_domain_range=20,
                        frequency_sample_prob=0.1,
                        sobel_exponential=2
                        ):

    """
    对img进行三角采样，并使用三角中的颜色进行填充，支持的颜色方法：
    center，取三角中心点;
    random，随机在三角形采样;
    vertex，取三角形点进行平均;

    采样点的方法：
    random，随机采样。
    fourier，用傅里叶变换对高频点和低频点分别采样。
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
    
    width = img.shape[0]
    height = img.shape[1]
    # Create an array of points.
    points = set({(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)}) # 存储采样的点
    
    if sample_point_method=='random':
        # generate points randomly
        for i in range(point_num):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            points.add((y,x))

    elif sample_point_method=='fourier':
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        """draw img
        plt.subplot(121),plt.imshow(img_gray, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.suptitle("Fourier Transform")
        plt.show()
        """

        rows, cols = img_gray.shape
        crow,ccol = rows//2 , cols//2
        frequency_domain_range = 20
        fshift[crow-frequency_domain_range:crow+frequency_domain_range, ccol-frequency_domain_range:ccol+frequency_domain_range] = 0
        
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)

        img_back = np.real(img_back)
        
        #add the point (value>0)
        x_cordinates, y_cordinates = np.where(img_back>0)

        # fourier high frequency points
        for x, y in zip(x_cordinates, y_cordinates):
            if np.random.rand()>1.0-frequency_sample_prob:# randomly filter some points
                points.add( ( int(y), int(x) ) )    #  or (x,y) ?

        # also some random points
        width = img.shape[0]
        height = img.shape[1]
        for i in range(point_num):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            points.add((y,x))

        # utils.save_img("imgs/high_freq.jpg", img_back)
    elif sample_point_method=='sobel':
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sobelX = cv2.Sobel(img_gray, -1, 1, 0)
        img_sobelY = cv2.Sobel(img_gray, -1, 0, 1)
        img_sobel = (img_sobelX**2 + img_sobelY**2).astype(np.float32)
        # img_sobel = np.sqrt(img_sobel)
        img_sobel = img_sobel**sobel_exponential
        prob = img_sobel - np.min(img_sobel)
        prob = prob / np.sum(img_sobel, axis=(0, 1))
        prob = prob.flatten()

        points_seperate = np.meshgrid(np.arange(img_gray.shape[0]), np.arange(img_gray.shape[1]))
        points = np.stack(points_seperate, axis=1)

        points = []
        choices = np.random.choice(a=np.arange(img_gray.shape[0] * img_gray.shape[1]), size=point_num, replace=False, p=prob)
        for p in choices:
            points.append((p // img.shape[0], p % img.shape[0]))
            
    else:
        raise NotImplementedError(f"Point Sampling Method \"{sample_point_method}\" is not implemented.")



    
    # # Insert points into subdiv
    for p in points :
        try:
            subdiv.insert(p)
        except:
            import pdb; pdb.set_trace()

    # Draw delaunay triangles
    draw_delaunay_blur( img, subdiv, method ) 

    is_draw_points=True
    if is_draw_points:
        # Draw points
        for p in points :
            draw_point(img, p, (0,0,255))

    np.true_divide(img, 255)

    return img

def DelaunayTriangulationBlur_4Channel(img, point_num=1000, method="center", \
                        sample_point_method="random", 
                        frequency_domain_range=20,
                        frequency_sample_prob=0.1,
                        sobel_exponential=2
                        ):

    """
    对img进行三角采样，并使用三角中的颜色进行填充，支持的颜色方法：
    center，取三角中心点;
    random，随机在三角形采样;
    vertex，取三角形点进行平均;

    采样点的方法：
    random，随机采样。
    fourier，用傅里叶变换对高频点和低频点分别采样。
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
    
    width = img.shape[0]
    height = img.shape[1]
    # Create an array of points.
    points = set({(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)}) # 存储采样的点
    
    mask = np.zeros(img.shape[:-1])
    
    if sample_point_method=='random':
        # generate points randomly
        # sampling fixed times
        for i in range(point_num):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            points.add((y,x))

    elif sample_point_method=='fourier':
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        """draw img
        plt.subplot(121),plt.imshow(img_gray, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.suptitle("Fourier Transform")
        plt.show()
        """

        rows, cols = img_gray.shape
        crow,ccol = rows//2 , cols//2
        frequency_domain_range = 20
        fshift[crow-frequency_domain_range:crow+frequency_domain_range, ccol-frequency_domain_range:ccol+frequency_domain_range] = 0
        
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)

        img_back = np.real(img_back)
        
        #add the point (value>0)
        x_cordinates, y_cordinates = np.where(img_back>0)

        # fourier high frequency points
        for x, y in zip(x_cordinates, y_cordinates):
            if np.random.rand()>1.0-frequency_sample_prob:# randomly filter some points
                points.add( ( int(y), int(x) ) )    #  or (x,y) ?

        # also some random points
        width = img.shape[0]
        height = img.shape[1]
        for i in range(point_num):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            points.add((y,x))

        # utils.save_img("imgs/high_freq.jpg", img_back)
    elif sample_point_method=='sobel':
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_sobelX = cv2.Sobel(img_gray, -1, 1, 0)
        img_sobelY = cv2.Sobel(img_gray, -1, 0, 1)
        img_sobel = (img_sobelX**2 + img_sobelY**2).astype(np.float32)
        # img_sobel = np.sqrt(img_sobel)
        img_sobel = img_sobel**sobel_exponential
        prob = img_sobel - np.min(img_sobel)
        prob = prob / np.sum(img_sobel, axis=(0, 1))
        prob = prob.flatten()

        points_seperate = np.meshgrid(np.arange(img_gray.shape[0]), np.arange(img_gray.shape[1]))
        points = np.stack(points_seperate, axis=1)

        points = []
        choices = np.random.choice(a=np.arange(img_gray.shape[0] * img_gray.shape[1]), size=point_num, replace=False, p=prob)
        for p in choices:
            points.append((p // img.shape[0], p % img.shape[0]))
            
    else:
        raise NotImplementedError(f"Point Sampling Method \"{sample_point_method}\" is not implemented.")



    
    # # Insert points into subdiv
    for p in points :
        # try:
        subdiv.insert([int(p[1]), int(p[0])])
        # except:
        #     import pdb; pdb.set_trace()

    # Draw delaunay triangles
    draw_delaunay_blur( img, subdiv, method, mask ) 

    is_draw_points=False
    if is_draw_points:
        # Draw points
        for p in points :
            draw_point(img, p, (0,0,255))

    img = np.true_divide(img, 255)

    mask = np.expand_dims(mask, axis=2)
    return np.concatenate((img, mask), axis=2)

# @nb.jit()

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
    
    width = img.shape[0]
    height = img.shape[1]
    # Create an array of points.
    points = set({(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)}) # 存储采样的点
    

    # 随机采点
    # while len(points) < 1000:
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