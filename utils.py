import cv2
import numpy as np

class Box():
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        return "box:" + str(self.x) + ', ' + str(self.y) + ', ' + str(self.width) + ', ' + str(self.height)

    def center(self, cast_to_int=False):
        if cast_to_int:
            return (int(self.x+self.width*0.5), int(self.y+self.height*0.5))    
        else:
            return (self.x+self.width*0.5, self.y+self.height*0.5)    

    def tl(self, cast_to_int=False):
        if cast_to_int:
            return (int(self.x), int(self.y))    
        else:
            return (self.x, self.y)    

    def br(self, cast_to_int=False):
        if cast_to_int:
            return (int(self.x+self.width), int(self.y+self.height))    
        else:
            return (self.x+self.width, self.y+self.height)
    
    def box_points(self,cast_to_int=False):
        tl = self.tl(cast_to_int)
        br = self.br(cast_to_int)
        return tl,(br[0],tl[1]),br,(tl[0],br[1])
        

    def min_max_bbox(self, cast_to_int=False):
        tl = self.tl(cast_to_int)
        br = self.br(cast_to_int)
        return (tl[0],tl[1],br[0],br[1])


    def ExpandBoxSquare(self, coef):
        new_size = self.width*coef

        x0 = self.x - (new_size - self.width)*0.5
        y0 = self.y - (new_size - self.height)*0.5

        return Box(x0, y0, new_size, new_size)


    def ExpandBox(self, coef):
        new_width = self.width * coef
        new_height = self.height * coef

        x0 = self.x - (new_width - self.width)*0.5
        y0 = self.y - (new_height - self.height)*0.5

        return Box(x0, y0, new_width, new_height)

    def SquareBox(self, mode=0):
        if mode == -1:
            pivot_size = min(self.width, self.height)
        elif mode == 0:
            pivot_size = (self.width + self.height)*0.5
        else:
            pivot_size = max(self.width, self.height)

        cX = self.x + self.width*0.5
        cY = self.y + self.height*0.5

        return Box(int(round(cX-pivot_size*0.5)),int(round(cY-pivot_size*0.5)),int(round(pivot_size)),int(round(pivot_size)))

def SmartCrop(src, dstRect):
    top = bottom = left = right = 0
    
    cropRect = Box(dstRect.x,dstRect.y,dstRect.width,dstRect.height)
    
    if(dstRect.x < 0) :
        left = -dstRect.x
        cropRect.x = 0
        cropRect.width -= left
        
    if(dstRect.y < 0) :
        top = -dstRect.y
        cropRect.y = 0
        cropRect.height -= top
    
    outMargin = src.shape[1]-(dstRect.x+dstRect.width)
    if(outMargin < 0) :
        right = -outMargin
        cropRect.width -= right
    
    outMargin = src.shape[0]-(dstRect.y+dstRect.height)
    if(outMargin < 0) :
        bottom = -outMargin
        cropRect.height -= bottom
    
    if(top>0 or bottom>0 or right>0 or left>0) :
        # dst = cv2.copyMakeBorder(src[int(round(cropRect.y)):int(round(cropRect.y+cropRect.height)), int(round(cropRect.x)):int(round(cropRect.x+cropRect.width))], int(round(top)), int(round(bottom)), int(round(left)), int(round(right)), cv2.BORDER_REPLICATE)
        dst = cv2.copyMakeBorder(src[int(round(cropRect.y)):int(round(cropRect.y+cropRect.height)), int(round(cropRect.x)):int(round(cropRect.x+cropRect.width))], int(round(top)), int(round(bottom)), int(round(left)), int(round(right)), cv2.BORDER_CONSTANT, value=(128,128,128))
    else :
        dst = src[int(round(cropRect.y)):int(round(cropRect.y+cropRect.height)), int(round(cropRect.x)):int(round(cropRect.x+cropRect.width))]

    return dst, top, bottom, left, right


def Resize(image, width = None, height = None, inter = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    if width is not None and height is not None:

        if width == w and  height == h:
            return image

        if inter is None:
            if width > w or height > h:
                inter = cv2.INTER_CUBIC
            else:
                inter = cv2.INTER_AREA

        return cv2.resize(image, (width,height), interpolation = inter)


    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        if height != h:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            dim = (w, h)

    # otherwise, the height is None
    else:
        if width != w:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (w, h)

    if inter is None:
        if r > 1.0:
            inter = cv2.INTER_CUBIC
        else:
            inter = cv2.INTER_AREA


    # resize the image
    if w != dim[0] or h != dim[1]:
        resized = cv2.resize(image, dim, interpolation = inter)
    else:
        return image

    # return the resized image
    return resized

def sobel(image):
    h = cv2.Sobel(image, cv2.CV_32F, 0, 1, -1)
    v = cv2.Sobel(image, cv2.CV_32F, 1, 0, -1)
    img = cv2.add(h, v)
    return img

def exclude_face(image, width):
    part_width = int((image.shape[1] - width) / 2)
    part1 = image[0:image.shape[0], 0:part_width]
    part2 = image[0:image.shape[0], image.shape[1] - part_width - 1:image.shape[1] - 1]
    img = np.concatenate((part1, part2), axis=1)
    return img