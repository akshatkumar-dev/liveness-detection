from mtcnn.mtcnn import MTCNN
import cv2
import os
from glob import glob
detector = MTCNN()
true_img_start = ('1', '2', 'HR_1')
def generate_frames_and_bbox(db_dir,save_dir,skip_num):
    file_list = open(save_dir+"/file_list.txt","a") #opening file to save the frame location and label if it is real or fake
    for file in glob("%s/*"%db_dir): #reading all files in the directory
        print("Processing video %s"%file)
        dir_name = os.path.join(save_dir, *file.replace(".mp4", "").split("/")[-3:]) #reading all mp4 files
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        frame_num = 0
        count = 0
        vidcap = cv2.VideoCapture(file) #capturing the frames
        success, frame = vidcap.read()
        while success:

            detect_res = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #detecting the face in the video
            if len(detect_res)>0 and count%skip_num==0:

                file_name = os.path.join(dir_name,"frame_%d.jpg" % frame_num) #getting the filename for the image frame
                

                label_txt = file.replace(".mp4", "").split("/")[-1][-1] #getting frame name

                label = 0 #setting label as 0 for fake image 1 for real image
                file_list.writelines("%s %d\n"%(file_name,label)) #saving the frame in the directory and writing the location in the file
                cv2.imwrite(file_name,frame)
                frame_num+=1 #increasing frame number
            count+=1
            success, frame = vidcap.read()

        vidcap.release()

    file_list.close()
def read():
    file = open("C:\\Users\\aksha\\Downloads\\Face-anti-spoofing-based-on-color-texture-analysis-master\\frames\\file_list.txt")
    for line in file:
        print(line.strip("\n").split(" "))


if __name__ == '__main__':
    db_dir = "C:\\Users\\aksha\\Downloads\\Face\\dataset\\rm\\train" #location of videos
    save_dir = "C:\\Users\\aksha\\Downloads\\Face" #location to save the frames from videos
    generate_frames_and_bbox(db_dir,save_dir,3)


    # read()
