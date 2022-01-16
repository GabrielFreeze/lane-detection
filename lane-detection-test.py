import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt
import copy
dir_path = os.path.dirname(os.path.realpath(__file__))

class Lane:
    def __init__(self,width,height):
        self.width = width
        self.height = height

        x_scl = 0.40
        y_scl = 0.55

        a = (0.00*self.width,      1.00*self.height)
        b = (x_scl*self.width,     y_scl*self.height)
        c = ((1-x_scl)*self.width, y_scl*self.height)
        d = (1.00*self.width,      1.00*self.height)

        self.mask_vertices = np.array([[a,b,c,d]], dtype=np.int32) 

        self.wrp_x1 = self.width/2 - self.width/10
        self.wrp_x2 = self.width/2 + self.width/10

        self.xm_in_px = 3.675 / 85 #Lane width (12 ft in m) is ~85 px on image
        self.ym_in_px = 3.048 / 24 #Dashed line length (10 ft in m) is ~24 px on image
        self.MAX_RADIUS = 10000    #Largest possible lane curve radius
        self.EQUID_POINTS = 25     #Number of points to use for the equidistant approximation
        self.DEV_POL = 2   #Max mean squared error of the approximation
        self.MSE_DEV = 1.1 #Minimum mean squared error ratio to consider higher order of the polynomial
        self.WINDOW_SIZE = 15 # Half of the sensor span
        self.DEV = 7 # Maximum of the point deviation from the sensor center
        self.SPEED = 2 / self.height # Pixels shift per frame
        self.RANGE = 0.0 # Fraction of the image to skip

        self.M, self.Minv = self.create_m()


    def set_gray(self, img):
       return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

    def get_roi(self, img):
        
        mask=np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, self.mask_vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    def eq_hist(self, img): # Histogram normalization
        return cv2.equalizeHist(img)

    def bin_thresh(self, img, param1=195, param2=255):
        _,frame = cv2.threshold(img,param1,param2,cv2.THRESH_BINARY)
        return frame

    def canny_edge(self, img, param1=0, param2=255):
        return cv2.Canny(img, param1, param2, 1)

    def get_lanes(self, edges):
        '''
            edges: A binary image of the result of a canny edge detection. 
            The image of the lanes should be rotated sideways (ie. lane lines are horizontal)

            returns 2 Lists of coordinates (x,y) for the points of the top and bottom lane lines, respectively
        '''
        try:
                                #y    ,    x
            pts_zip = list(zip(-pts[0],pts[1]))
            top_mostY = max(-pts[0])
            bottom_mostY = min(-pts[0])

            #Calculate horizontal line that seperates right and left lane
            midY = (top_mostY + bottom_mostY)/2

            top_ptsX = []
            top_ptsY = []

            bottom_ptsX = []
            bottom_ptsY = []
            
            #Put points in the correct array
            for y,x in pts_zip:
                if y > midY:
                    top_ptsX.append(x)
                    top_ptsY.append(y+edges.shape[0])
                else:
                    bottom_ptsX.append(x)
                    bottom_ptsY.append(y+edges.shape[0])
            
            top    = list(zip(top_ptsX,top_ptsY))
            bottom = list(zip(bottom_ptsX,bottom_ptsY))

            return (top,bottom)

        except Exception as e: 
            print(str(e))
            return [],[]

    def filter_points(self, pts, dev=1.6):
        '''
        pts: A list of coordinates (x,y)
        dev: The maximum deviation the filtered points should have
        returns a subset of pts such that any point's y coordinate's standard deviation does not exceed dev
        '''

        #Filter top lines to remove outlier points (noise)
        pts_y = [y for (x,y) in pts]
        mean = np.mean(pts_y)
        std = np.std(pts_y)
        return [(x,y) for (x,y) in pts if abs((y-mean)/std) < dev]

    def fit_curve(self, pts, order=2):
        '''
        pts: A list of coordinates (x,y)
        order: The Order the fitted curve should be
        returns a np.polyfit vector of coefficients
        '''
        
        #Fit a quadratic polynomial curve
        return np.polyfit(top_ptsXf, top_ptsYf, order)
    
    def draw_curve(self, img, pol, pts):
        '''
        img: The image to show the curve on.
        pol: An np.polyfit vector of coefficients
        pts: The points used to fit pol
        '''
        pts_x = [x for (x,_) in pts]

        x = np.arange(min(pts_x),max(pts_x),1)
        y_org = np.polyval(pol,x)
        y = [rotated_canny_edges.shape[0]-i for i in y_org]
        pts = np.array(list(zip(x,y)), np.int32)

        return cv2.polylines(img, [pts], False, (0,0,255),3)


    #Returns saturation channel of img
    def s_hls(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return hls[:,:,2]

    # Sharpen image
    def sharpen_img(self, img):
        gb = cv2.GaussianBlur(img, (5,5), 20.0)
        return cv2.addWeighted(img, 2, gb, -1, 0)

    # Compute linear image transformation img*s+m
    def lin_img(self, img,s=1.0,m=0.0):
        img2=cv2.multiply(img, np.array([s]))
        return cv2.add(img2, np.array([m]))

    # Change image contrast; s>1 - increase
    def contr_img(self, img, s=1.0):
        m=127.0*(1.0-s)
        return self.lin_img(img, s, m)

    # Create perspective image transformation matrices
    def create_m(self, ):
        src = np.float32([[0, self.height],[self.width,self.height],[0,0.625*self.height],[self.width,0.625*self.height]])
        dst = np.float32([[self.wrp_x1,self.height],[self.wrp_x2,self.height],[0,0],[self.width,0]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    # Main image transformation routine to get a warped image
    def transform(self, img, M):
        img_size = (self.width, self.height)
        img = cv2.warpPerspective(img, M, img_size)
        img = self.sharpen_img(img)
        img = self.contr_img(img, 1.1)
        return img

    #Calculate coefficients of a polynomial in y+h coordinates, i.e. f(y) -> f(y+h)
    
    def pol_shift(self, pol, h):
        pol_ord = len(pol)-1 # Determinate degree of the polynomial 
        if pol_ord == 3:
            pol0 = pol[0]
            pol1 = pol[1] + 3.0*pol[0]*h
            pol2 = pol[2] + 3.0*pol[0]*h*h + 2.0*pol[1]*h
            pol3 = pol[3] + pol[0]*h*h*h + pol[1]*h*h + pol[2]*h
            return(np.array([pol0, pol1, pol2, pol3]))
        if pol_ord == 2:
            pol0 = pol[0]
            pol1 = pol[1] + 2.0*pol[0]*h
            pol2 = pol[2] + pol[0]*h*h+pol[1]*h
            return(np.array([pol0, pol1, pol2]))
        if pol_ord == 1:
            pol0 = pol[0]
            pol1 = pol[1] + pol[0]*h
            return(np.array([pol0, pol1]))

    # Calculate derivative for a polynomial pol in a point x
    def pol_d(self, pol, x):
        pol_ord = len(pol)-1
        if pol_ord == 3:
            return 3.0*pol[0]*x*x+2.0*pol[1]*x+pol[2]
        if pol_ord == 2:
            return 2.0*pol[0]*x+pol[1]
        if pol_ord == 1:
            return pol[0]#*np.ones(len(np.array(x)))
        
    # Calculate the second derivative for a polynomial pol in a point x
    def pol_dd(self, pol, x):
        pol_ord = len(pol)-1
        if pol_ord == 3:
            return 6.0*pol[0]*x+2.0*pol[1]
        if pol_ord == 2:
            return 2.0*pol[0]
        if pol_ord == 1:
            return 0.0
        
    # Calculate a polinomial value in a given point x
    def pol_calc(self, pol, x):
        pol_f = np.poly1d(pol)
        return(pol_f(x))

    def px_to_m(self, px): # Conver ofset in pixels in x axis into m
        return self.xm_in_px*px

    # Calculate offset from the lane center
    def lane_offset(self, left, right):
        offset = self.width/2.0-(self.pol_calc(left, 1.0) + self.pol_calc(right, 1.0))/2.0
        return self.px_to_m(offset)

    # Calculate radius of curvature of a line
    def r_curv(self, pol, y):
        if len(pol) == 2: # If the polinomial is a linear function
            return self.MAX_RADIUS
        else:
            y_pol = np.linspace(0, 1, num=self.EQUID_POINTS)
            x_pol = self.pol_calc(pol, y_pol)*self.xm_in_px
            y_pol = y_pol*self.height*self.ym_in_px
            pol = np.polyfit(y_pol, x_pol, len(pol)-1)
            d_y = self.pol_d(pol, y)
            dd_y = self.pol_dd(pol, y)
            r = ((np.sqrt(1+d_y**2))**3)/abs(dd_y)
            if r > self.MAX_RADIUS:
                r = self.MAX_RADIUS
            return r
    
    # Calculate radius of curvature of a lane by avaraging lines curvatures
    def lane_curv(self, left, right):
        l = self.r_curv(left, 1.0)
        r = self.r_curv(right, 1.0)
        if l < self.MAX_RADIUS and r < self.MAX_RADIUS:
            return (self.r_curv(left, 1.0)+self.r_curv(right, 1.0))/2.0
        else:
            if l < self.MAX_RADIUS:
                return l
            if r < self.MAX_RADIUS:
                return r
            return self.MAX_RADIUS

    def mean_squared_error(self,true,pred):
        return np.square(np.subtract(true,pred)).mean()

    # Choose the best polynomial order to fit points (x,y)
    def best_pol_ord(self, x, y):
        pol1 = np.polyfit(y,x,1)
        pred1 = self.pol_calc(pol1, y)
        mse1 = self.mean_squared_error(x, pred1)

        if mse1 < self.DEV_POL:
            return pol1, mse1

        pol2 = np.polyfit(y,x,2)
        pred2 = self.pol_calc(pol2, y)
        mse2 = self.mean_squared_error(x, pred2)

        if mse2 < self.DEV_POL or mse1/mse2 < self.MSE_DEV:
                return pol2, mse2
        else:
            pol3 = np.polyfit(y,x,3)
            pred3 = self.pol_calc(pol3, y)
            mse3 = self.mean_squared_error(x, pred3)
            return pol2, mse2 if mse2/mse3 < self.MSE_DEV else pol3, mse3
    
    # Smooth polynomial functions of different degrees   
    def smooth_dif_ord(self, pol_p, x, y, new_ord):
        x_p = self.pol_calc(pol_p, y)
        x_new = (x+x_p)/2.0
        return np.polyfit(y, x_new, new_ord)


# frame_org = cv2.imread(dir_path+"/lane2.png")


cv2.namedWindow('Binary Threshold')
cv2.createTrackbar('Binary Threshold', 'Binary Threshold', 226, 255, lambda x: None)

cv2.namedWindow('Canny Threshold')
cv2.createTrackbar('Canny Threshold', 'Canny Threshold', 50, 200, lambda x: None)


cap = cv2.VideoCapture(dir_path+"/lane-test5.mp4")
_,frame_org = cap.read()
lane = Lane(frame_org.shape[1], frame_org.shape[0])
while cap.isOpened():

    _,frame_org = cap.read()
    frame = cv2.cvtColor(frame_org, cv2.COLOR_RGB2GRAY)
    frame = cv2.equalizeHist(frame)
    
    # _,frame = cv2.threshold(frame,195,255,cv2.THRESH_BINARY)
    _,frame = cv2.threshold(frame,cv2.getTrackbarPos('Binary Threshold','Binary Threshold'),255,cv2.THRESH_BINARY)

    warped_frame = lane.get_roi(frame)
    warped_frame = lane.transform(warped_frame,lane.M)

    # canny_edges = cv2.Canny(warped_frame, 50,200, 1)
    canny_edges = cv2.Canny(warped_frame, cv2.getTrackbarPos('Canny Threshold','Canny Threshold'),200, 1)
    # cv2.imshow('!',canny_edges)

    frame2 = frame_org.copy()
    frame2 = lane.transform(frame2, lane.M)
    frame2 = cv2.rotate(frame2,cv2.ROTATE_90_CLOCKWISE)
    
    #Rotate Image
    rotated_canny_edges = cv2.rotate(canny_edges, cv2.ROTATE_90_CLOCKWISE)
    
    pts = np.where(rotated_canny_edges != [0])
    
    try:
                    #y,x
        pts_zip = list(zip(-pts[0],pts[1]))
        top_mostY = max(-pts[0])
        bottom_mostY = min(-pts[0])

        #Calculate horizontal line that seperates right and left lane
        midY = (top_mostY + bottom_mostY)/2
        print(top_mostY+rotated_canny_edges.shape[0],bottom_mostY+rotated_canny_edges.shape[0],midY+rotated_canny_edges.shape[0])

        top_ptsX = []
        top_ptsY = []

        bottom_ptsX = []
        bottom_ptsY = []
        
        #Put points in the correct array
        for y,x in pts_zip:
            if y > midY:
                top_ptsX.append(x)
                top_ptsY.append(y+rotated_canny_edges.shape[0])
            else:
                bottom_ptsX.append(x)
                bottom_ptsY.append(y+rotated_canny_edges.shape[0])

        #Filter top lines to remove outlier points (noise)
        mean = np.mean(top_ptsY)
        std = np.std(top_ptsY)
        top_ptsXf = []
        top_ptsYf = []

        for i,y in enumerate(top_ptsY):
            if abs((y-mean)/std) < 1.6:
                top_ptsXf.append(top_ptsX[i])
                top_ptsYf.append(y)
        

        #Filter bottom lines to remove outlier points (noise)
        mean = np.mean(bottom_ptsY)
        std = np.std(bottom_ptsY)
        bottom_ptsXf = []
        bottom_ptsYf = []

        for i,y in enumerate(bottom_ptsY):
            if abs((y-mean)/std) < 1.6:
                bottom_ptsXf.append(bottom_ptsX[i])
                bottom_ptsYf.append(y)



        #Fit a quadratic polynomial curve
        zl = np.polyfit(top_ptsXf, top_ptsYf, 2)
        fl = np.poly1d(zl)

        zr = np.polyfit(bottom_ptsXf, bottom_ptsYf, 2)
        fr = np.poly1d(zr)



        # fig, axs = plt.subplots(2)
        # axs[0].scatter(top_ptsXf,top_ptsYf)
        # plt.ylim([min(top_ptsYf),max(top_ptsYf)])
        # t = np.arange(min(top_ptsXf),max(top_ptsXf),1)
        # axs[1].plot(t,fl(t))
        # plt.show()

        # fig, axs = plt.subplots(2)
        # axs[0].scatter(bottom_ptsXf,bottom_ptsYf)
        # plt.ylim([min(bottom_ptsYf),max(bottom_ptsYf)])
        # t = np.arange(min(bottom_ptsXf),max(bottom_ptsXf),1)
        # axs[1].plot(t,fr(t))
        # plt.show()

        #Plot curve on lane video
        x = np.arange(min(top_ptsXf),max(top_ptsXf),1)
        y_org = np.polyval(zl,x)
        y = [rotated_canny_edges.shape[0]-i for i in y_org]
        pts = np.array(list(zip(x,y)), np.int32)
        cv2.polylines(frame2, [pts], False, (0,0,255),3)

        #Plot curve on lane video
        x = np.arange(min(bottom_ptsXf),max(bottom_ptsXf),1)
        y_org = np.polyval(zr,x)
        y = [rotated_canny_edges.shape[0]-i for i in y_org]
        pts = np.array(list(zip(x,y)), np.int32)
        cv2.polylines(frame2, [pts], False, (0,0,255),3)
    except Exception as e:  print("Could not perform lane detection"+str(e))

    #Show frame
    frame2 = cv2.rotate(frame2,cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow('frame2',frame2)

    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
        break
    



    # fig, axs = plt.subplots(2)
    # axs[0].scatter(top_ptsX,top_ptsY)
    # t = np.arange(min(top_ptsX),max(top_ptsX),1)
    # axs[1].plot(t,fl(t))

    # plt.show()


    


    # frame2 = lane.transform(frame2, lane.Minv)
    # cv2.imshow('canny_edges', frame2)

        

# plt.xlim([0,lane.width])
# # plt.ylim([0,-lane.width])
# plt.scatter(data[0],data[1])
# plt.show()


# z = np.polyfit(data[0], data[1], 5)
# f = np.poly1d(z)
# t = np.arange(0, canny_edges.shape[1], 1)
# plt.plot(t,f(t))
# plt.show()















    # frame = lane.eq_hist(frame)
    # frame = frame[:,:,2]+0.5*lane.s_hls(frame)
    # frame = np.uint8(255*frame/np.max(frame))
    # lines = cv2.HoughLines(frame, 1, np.pi / 180, 150, None, 0, 0)

    # if lines is not None:
    #     for l in lines:
    #         rho = l[0][0]
    #         theta = l[0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    #         if (pt1[0]-pt2[0]) != 0:
    #             m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
                
    #             if m > 10 or m < -10:
    #                 cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)