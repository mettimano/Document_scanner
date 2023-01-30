import cv2
import numpy as np
import pytesseract
import sys
from pathlib import PurePath


def get_canny_parameters(image):

  #Get the median value of the image
  v = np.median(image)

  #Produce the upper and lower bounds  
  lower_bound = int(0.75* v)
  upper_bound = int(min(255,1.25*  v))

  return lower_bound,upper_bound

def get_document_contour(contours):
    document = np.array([])
    max_area = 0

    #checking all contours
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.015 * peri, True)

        #Saving countor with biggest area and 4 sides
        if area > max_area and len(approx) == 4:
            document = approx
            max_area = area

    return document

def get_warped_img(img,points):
  rect_points = np.zeros((4, 2), dtype="float32")
  #Document corners
  points_sum = points.sum(axis=1)
  rect_points[0] = points[np.argmin(points_sum)]
  rect_points[3] = points[np.argmax(points_sum)]

  points_diff = np.diff(points, axis=1)
  rect_points[1] = points[np.argmin(points_diff)]
  rect_points[2] = points[np.argmax(points_diff)]

  (tl, tr, br, bl) = rect_points

  #Document dimensions
  top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

  max_width = max(int(bottom_width), int(top_width))
  max_height = max(int(right_height),int(left_height))

 
  destination_points = np.float32([
    [0, 0], 
    [max_width -1, 0], 
    [0, max_height-1], 
    [max_width-1, max_height-1]
    ])
  
  #Perspective transform matrix
  matrix = cv2.getPerspectiveTransform(rect_points, destination_points)

  #Warp
  warped_img = cv2.warpPerspective(img, matrix, (max_width, max_height))

  return warped_img

def scan_img(img_path):
  #Load img
  img = cv2.imread(img_path)
  original_img = img.copy()

  # Image filter
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  filtered_gray = cv2.GaussianBlur(gray, (3,3), 3)
  #Edge detection
  lower_bound,upper_bound=get_canny_parameters(filtered_gray)
  edged = cv2.Canny(filtered_gray,lower_bound,upper_bound)


  # Contour detection
  contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  document_contour=get_document_contour(contours)

  #Warp
  warped_img=get_warped_img(original_img,document_contour.reshape(4,2))

  return warped_img

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return threshold

def main():


  #Getting the image path as argument
  img_path=sys.argv[1]
  action=sys.argv[2]

  #Getting the paht of the image's folder
  path=PurePath(img_path)
  path_output=path.parents[0]

  #getting the warped image that centers the document
  scanned_img=scan_img(img_path)

  #giving the image more contrast
  processed = image_processing(scanned_img)

  #action chosen by the user
  if action == 's':
      cv2.imwrite(str(path_output)+"/scanned_output.jpg", processed)
      
  elif action == 'o':
      file = open(str(path_output)+"/recognized_output.txt", "w")
      ocr_text = pytesseract.image_to_string(scanned_img)
      file.write(ocr_text)
      file.close()

if __name__ == "__main__":
    main()
