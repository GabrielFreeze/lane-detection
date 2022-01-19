import numpy as np
import cv2
import os
import math
import time
from matplotlib import pyplot as plt
import copy
dir_path = os.path.dirname(os.path.realpath(__file__))+'/'

class Lane:
    def __init__(self, width, height, x_scl=0.4, y_scl=0.55):
        self.width = width
        self.height = height

        a = (0.00*self.width,      1.00*self.height)
        b = (x_scl*self.width,     y_scl*self.height)
        c = ((1-x_scl)*self.width, y_scl*self.height)
        d = (1.00*self.width,      1.00*self.height)

        self.mask_vertices = np.array([[a,b,c,d]], dtype=np.int32) 

        self.wrp_x1 = self.width/2 - self.width/10
        self.wrp_x2 = self.width/2 + self.width/10


        self.min_lane_pts = 175         #Minimum Number of Points a detected lane line should contain.
                                        #Less than that, then it is considered noise.
        self.shift = 20                 #The estimated distance in px between the 2 lanes. Used for lane inference
        self.MAX_RADIUS = float('inf')  #Largest possible lane curve radius
        self.EQUID_POINTS = 25          #Number of points to use for the equidistant approximation
        self.DEV = 7                    # Maximum of the point deviation from the sensor center

        self.M, self.Minv = self.create_m()
    
    def set_roi(self, x_scl, y_scl):
        a = (0.00*self.width,      1.00*self.height)
        b = (x_scl*self.width,     y_scl*self.height)
        c = ((1-x_scl)*self.width, y_scl*self.height)
        d = (1.00*self.width,      1.00*self.height)

        self.mask_vertices = np.array([[a,b,c,d]], dtype=np.int32) 

    def set_min_lane_pts(self, x):
        self.min_lane_pts = x;

    def set_gray(self, img):
       return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

    def set_shift(self,x):
        self.shift = x

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
            pts = np.where(edges != [0])
                                #y    ,    x
            pts_zip = list(zip(-pts[0],pts[1]))
            top_mostY = max(-pts[0])
            bottom_mostY = min(-pts[0])

            #Calculate horizontal line that seperates right and left lane
            # midY = (top_mostY + bottom_mostY)/2
            midY = -self.width/2

            top_ptsX = []
            top_ptsY = []
            top = []

            bottom_ptsX = []
            bottom_ptsY = []
            bottom = []
            
            #Put points in the correct array
            for y,x in pts_zip:
                if y > midY:
                    top_ptsX.append(x)
                    top_ptsY.append(y+edges.shape[0])
                else:
                    bottom_ptsX.append(x)
                    bottom_ptsY.append(y+edges.shape[0])
            
            if len(top_ptsX) > self.min_lane_pts:
                top = list(zip(top_ptsX,top_ptsY))

            if len(bottom_ptsY) > self.min_lane_pts:
                bottom = list(zip(bottom_ptsX,bottom_ptsY))


            # print(len(top_ptsX),len(bottom_ptsX))

            return (top,bottom)

        except Exception as e: 
            print(str(e))
            return [],[]

    def filter_points(self, pts, dev=1.6):
        '''
        pts: A list of coordinates (x,y)
        dev: The maximum deviation the filtered points should have
        
        returns a subset of pts such that any point's y coordinate's standard deviation that is not considered noise.
        '''

        #If there aren't enough points, treat the detected points as noise.
        if len(pts) < self.min_lane_pts:
            return []

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
        return np.polyfit([x for (x,_) in pts], [y for (_,y) in pts], order)
    
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

        return cv2.polylines(img, [pts], False, (0,0,255),10)

    def infer_lane(self, pol, pts, other_lane):
        '''
        pol: A np.polyfit vector representing the curve to be copied.
        pts: A list or coordinates (x,y) of the points used to generate pol.
        other_lane: True if pol is the curve for the left lane, false for the right.
        returns: the new set of points, an np.polyfit vector copied and translated from pol
        '''

        #Get average y coordinated of pts
        new_pol = [0]*3
        y_coords = [y for (_,y) in pts]
        avg_y = sum(y_coords)/len(pts)
        
        # shift = self.width/2 - avg_y if other_lane else avg_y - self.width/2
        shift = -self.shift if other_lane else self.shift
        #Quadratic Curve
        if len(pol) == 3:
            new_pol[0] = pol[0]
            new_pol[1] = pol[1]      
            new_pol[2] = pol[2]+shift #Move curve up/left if other_lane lane is false (Left) or down/right if other_lane is true (left)
        
        #Generate new points
        f = np.poly1d(pol)
        t = np.arange(0, max(y_coords), 5)
        new_pts = [(x,f(x)) for x in t]

        return new_pts,new_pol
    
    def _infer_lane(self, pol, k, other_lane):
        assert(len(pol) == 3)
        a = pol[0]
        b = pol[1]
        c = pol[2]

        x = (-b)/(2*a)              # x Coordinate of Turning Point of pol
        y = self.pol_calc(pol,x)    # y Coordinate of Turning Point of pol

        r = (math.sqrt(1 + 4*(a**2)*(x**2) + 4*a*b*x + b**2 )**3)/(2*a) # Radius of curvature at Turning Point

        p = a > 0 #True: Curve is ∪ shaped. False: Curve is ∩ shaped.

        y_center = y+r if p else y-r #Center of circle

        y_center_new = (y_center)-r*k if p else (y_center)+r*k #Center of scaled circle

        r_new = r - r*k  #Radius of scaled circle

        #Find y-point on new circle at x, ±2. 
        y2 = y_center_new + r_new

        #Find Quadratic Curve that fits the turning point, and the 2 new adjacent points of this circle
        w = 20
        curve = np.polyfit([x-w,x,x+w],[y2,y,y2],2)

        # curve[2] += -k*200 if other_lane else k*200

        return list(zip([x-w,x,x+w],[y2,y,y2])), curve

    def remove_horizontal(self, img, k=1, stroke = 10):
        '''
        Removes any horizontal lines from the binary image img with a gradient in the range [min_m,max_m]
        returns: The filtered image
        '''
        lines = cv2.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for l in lines:
                rho = l[0][0]
                theta = l[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                if (pt1[0]-pt2[0]) != 0:
                    m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
                    
                    if -k < m < k:
                        cv2.line(img, pt1, pt2, (0,0,0), stroke, cv2.LINE_AA)
        return img

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
            x_pol = self.pol_calc(pol, y_pol)
            y_pol = y_pol*self.height
            pol = np.polyfit(y_pol, x_pol, len(pol)-1)
            d_y = self.pol_d(pol, y)
            dd_y = self.pol_dd(pol, y)
            r = ((np.sqrt(1+d_y**2))**3)/abs(dd_y)
            if r > self.MAX_RADIUS:
                r = self.MAX_RADIUS
            return r,(dd_y > 0)
    
    # Calculate radius of curvature of a lane by avaraging lines curvatures
    def lane_curv(self, left, right):
        l,dir_l = self.r_curv(left, 1.0)
        r,dir_r = self.r_curv(right, 1.0)
        if l < self.MAX_RADIUS and r < self.MAX_RADIUS:
            return (l+r)/2.0, (dir_l and dir_r)
        else:
            if l < self.MAX_RADIUS:
                return l, False
            if r < self.MAX_RADIUS:
                return r, False
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


def show_image(name, img, size=0.35):
    cv2.imshow(name,cv2.resize(img,(int(img.shape[1]*size),int(img.shape[0]*size))))

#TODO: Hough Line transform to remove horizontal lines from the binary image

cap = cv2.VideoCapture(dir_path+"lane-test2.mp4")
# _,frame_org = cap.read()
frame_org = cv2.imread(dir_path+"lane-test12.png")
lane = Lane(frame_org.shape[1], frame_org.shape[0])


cv2.namedWindow('Hyper Parameters')
cv2.createTrackbar('Binary Threshold',       'Hyper Parameters', 245, 255,    lambda x: None)
cv2.createTrackbar('Canny Threshold',        'Hyper Parameters', 50,  200,    lambda x: None)
cv2.createTrackbar('Y-Scale',                'Hyper Parameters', 71,  100,    lambda x: None)
cv2.createTrackbar('X-Scale',                'Hyper Parameters', 20,  100,    lambda x: None)
cv2.createTrackbar('Minimum STD',            'Hyper Parameters', 16,  100,    lambda x: None)
cv2.createTrackbar('Minimum Lane Pts',       'Hyper Parameters', 175, 1000,   lambda x: None)
cv2.createTrackbar('Distance between Lanes', 'Hyper Parameters', 625,   1000,   lambda x: None)
cv2.createTrackbar('Horizontal Gradient Range:', 'Hyper Parameters', 1, 200,   lambda x: None)
cv2.createTrackbar('Horizontal Stroke:', 'Hyper Parameters', 1, 50,   lambda x: None)



while cap.isOpened():


    lane.set_roi(cv2.getTrackbarPos('X-Scale','Hyper Parameters')/100,cv2.getTrackbarPos('Y-Scale','Hyper Parameters')/100)
    lane.set_min_lane_pts(cv2.getTrackbarPos('Minimum Lane Pts','Hyper Parameters'))
    lane.set_shift(cv2.getTrackbarPos('Distance between Lanes', 'Hyper Parameters'))

    # _,frame_org = cap.read()
    frame_org = cv2.imread(dir_path+"lane-test12.png")
    show_image('Original Frame',lane.get_roi(frame_org))

    frame = lane.set_gray(frame_org)
    frame = lane.eq_hist(frame)

    frame = lane.bin_thresh(frame,param1=cv2.getTrackbarPos('Binary Threshold','Hyper Parameters'),param2=255)

    warped_frame = lane.get_roi(frame)
    warped_frame = lane.transform(warped_frame,lane.M)
    
    # show_image('Warped Frame',cv2.line(warped_frame,(lane.width//2,0),(lane.width//2,lane.height),(255,255,255),3))

    frame2 = frame_org.copy()
    frame2 = lane.transform(frame2, lane.M)
    frame2 = cv2.rotate(frame2,cv2.ROTATE_90_CLOCKWISE)
    

    warped_frame = lane.remove_horizontal(warped_frame,
                                        cv2.getTrackbarPos('Horizontal Gradient Range:', 'Hyper Parameters')/100,
                                        cv2.getTrackbarPos('Horizontal Stroke:', 'Hyper Parameters'))

    show_image('Warped Frame',warped_frame)
    canny_edges = lane.canny_edge(warped_frame, param1=cv2.getTrackbarPos('Canny Threshold','Hyper Parameters'), param2=200)

    #Rotate Image
    rotated_canny_edges = cv2.rotate(canny_edges, cv2.ROTATE_90_CLOCKWISE)
    
    left,right = lane.get_lanes(rotated_canny_edges)
    left_curve,right_curve = [],[]

    # plt.scatter([x for (x,y) in left],[y for (x,y) in left])
    # plt.scatter([x for (x,y) in right],[y for (x,y) in right])
    # plt.show()

    try:

        std_dev = cv2.getTrackbarPos('Minimum STD','Hyper Parameters')/10

        if left and (left_f := lane.filter_points(left,std_dev)):
            left_curve = lane.fit_curve(left_f)
            if not right:
                print('Inferring Right Lane ['+str(time.time()%32)+']')
                right_f,right_curve = lane.infer_lane(left_curve,left,True)
                # right_f,right_curve = lane._infer_lane(left_curve,0.2,True)
                # plt.scatter([x for x,y in left_f], [y for x,y in left_f])
                # plt.scatter([x for x,y in right_f], [y for x,y in right_f])
                # plt.show()
                
        if right and (right_f := lane.filter_points(right,std_dev)):
            right_curve = lane.fit_curve(right_f)
          
            if not left:
                print('Inferring Left Lane ['+str(time.time()%32)+']')
                left_f,left_curve = lane.infer_lane(right_curve,right,False)
                # left_f,left_curve = lane._infer_lane(right_curve,0.2,False)
        
        if len(left_curve) and len(right_curve):
            frame2 = lane.draw_curve(frame2, left_curve,  left_f)
            frame2 = lane.draw_curve(frame2, right_curve, right_f)
        else:
            #No lanes were detected
            #Do something?
            pass



        # if int(time.time()*32) % 2 == 0:
        #     x,dir = lane.lane_curv(left_curve, right_curve)
        #     if x/100000000 < 100:
        #         print(x/100000000,['RIGHT','LEFT'][dir])
        #     else: print('Straight')
            
    except Exception as e: print(str(e))
    

    #Show frame
    frame2 = cv2.rotate(frame2,cv2.ROTATE_90_COUNTERCLOCKWISE)
    show_image('Output Frame',frame2)

    
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