import cv2
import numpy as np
import matplotlib.pyplot as plt
x,y,k=200,200,-1

capture=cv2.VideoCapture(0)

def take_input_images(action_event,x1,y1,flag,param):
    global x,y,k
    if action_event==cv2.EVENT_LBUTTONDOWN:
        x=x1
        y=y1
        k=1

def set_window_details():
	cv2.namedWindow("Draw/Move_point")
	cv2.setMouseCallback("Draw/Move_point",take_input_images)


set_window_details()
while(True):
    _,input_image=capture.read()
    input_image=cv2.flip(input_image,1)
    grayscale_input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Draw/Move_point",input_image)
    if(k==1 or cv2.waitKey(30)==27):
        cv2.destroyAllWindows()
        break

stp=0

old_points=np.array([[x,y]],dtype=np.float32).reshape(-1,1,2)

mask=np.zeros_like(input_image)

while(True):
    _, new_inp_img = capture.read()
    new_inp_img = cv2.flip(new_inp_img, 1)
    new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)     
    new_pts,status,err = cv2.calcOpticalFlowPyrLK(grayscale_input_image, 
                         new_gray, 
                         old_points, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))

    for i, j in zip(old_points, new_pts):
        x,y = j.ravel()
        a,b = i.ravel()
        if cv2.waitKey(2) & 0xff == ord('q'):
            stp = 1
            
        elif cv2.waitKey(2) & 0xff == ord('w'):
            stp = 0
        
        elif cv2.waitKey(2) == ord('n'):
            mask = np.zeros_like(new_inp_img)
            
        if stp == 0:
            mask = cv2.line(mask, (a,b), (x,y), (0,0,255), 6)

        cv2.circle(new_inp_img, (x,y), 6, (0,255,0), -1)
    
    new_inp_img = cv2.addWeighted(mask, 0.3, new_inp_img, 0.7, 0)
    cv2.putText(mask, "'q' to gap 'w' - start 'n' - clear", (10,50), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
    cv2.imshow("ouput", new_inp_img)
    cv2.imshow("result", mask)

    
    grayscale_input_image = new_gray.copy()
    old_points = new_pts.reshape(-1,1,2)
    
    if cv2.waitKey(1) & 0xff == 27:
        break

cv2.destroyAllWindows()
capture.release()
