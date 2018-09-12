import pymongo
import numpy as np
from gridfs import GridFS
from PIL import Image as pil
from io import BytesIO
import scipy.misc
import pickle
import cv2

class Image:
    def getConnection(self,db,col,uri):

        self.cl=pymongo.MongoClient(uri)
        self.db=self.cl[db]
        self.col=self.db[col]
        self.idx = 0

    def getImage(self):

        check=self.col.find({'image.v.chips':{'$exists':True}}).limit(1000)
        # ImageData= pickle.loads(check['annotations']['pixellabel']['PVS']['image'])
        # image=np.array(pil.open(BytesIO(ImageData)))
        fs = GridFS( self.db,'image_data')
        for doc in check:
          print(doc)
          hashkey=doc['image']['raw']['hash']
          #boxlabel= doc['annotations']['boxlabel']['[0]']['x0']
          for box in doc["annotations"]["boxlabel"]:
                 #x0 = box["x0"]
                 x0=box["x0"]
                 y0=box["y0"]
                 x1=box["x1"]
                 y1=box["y1"]
                 print(x1-x0)
                 print(y1-y0)
                 self.writeTofile(x0,y0,x1-x0,y1-y0)
                 break

          #ImageData = np.asarray(pil.open(BytesIO(fs.get(hashkey).read()))).astype(np.uint32)
        # self.getCutOuts(ImageData,hashkey)
        # self.saveImage(ImageData,"orignial")

          #image_name='%06d' %(self.idx + 1)
          #self.idx+=1
          #self.saveImage(ImageData,image_name)
        # self.getCutOuts(ImageData,hashkey)

    def saveToPgm(self, ImageData, fileName):
       file=open('D:\\'+fileName+'.pgm','w')
       file.write("P5")
       file.write("\n")

       file.write(str(ImageData.shape[0]))
       file.write("\t")
       file.write(str(ImageData.shape[1]))
       file.write("\n")
       file.write("4095")
       file.write("\n")
       for x in range(ImageData.shape[0]):
            file.write(str(ImageData[x,y]))
            file.write(" ")
            file.write("\n")

       file.close()

    def saveImage(self,ImageData,fileName):
        fileName='D:\\'+fileName
        scipy.misc.imsave(fileName+'.jpg', ImageData)
        # cv2.imwrite('D:\\'+fileName+'.pgm', ImageData)

    def writeTofile(self,x0,y0,x1,y1):
        f = open("demofile.txt", "a")
        #string1=str(x0)+","+str(y0)+","+str(x1)+","+str(y1)+"\n"
        f.write("%s" %x0)
        f.write(" ")
        f.write("%s" %y0)
        f.write(" ")
        f.write("%s" %x1)
        f.write(" ")
        f.write("%s" %y1)
        f.write("\n")

""" def getCutOuts(self,img_data,hashkey):
         data=self.col.find_one({'image.y.chips.hash':hashkey})
         in_data=data['annotations']

         for boxes in range(len(in_data['boxlabel'])):

            left, right, top, bottom = int(in_data['boxlabel'][boxes]["x0"]), int(
                in_data['boxlabel'][boxes]["x1"]), int(in_data['boxlabel'][boxes]["y0"]), int(
                in_data['boxlabel'][boxes]["y1"])
            if ((left == right) or (top == bottom) or left < 0 or top < 0 or right > 1175 or bottom > 481):
                continue
            cutout = img_data[top:bottom, left:right]
            # if (len(cutout) == 0):
            #     continue
            # label = in_data['boxlabel'][boxes]['attributes']['sign_class']['value']
            # if not any(d['LABEL_NAME'] == label for d in app.metamodel.class_list):
            #     continue
            # indx = next((index for (index, d) in enumerate(app.metamodel.class_list) if d["LABEL_NAME"] == label),
            #             None)
            # class_path = self.sign_class_database + "\\" + label + "\\" + self.real_database + "\\" + str(
            #     count_samples[indx]).zfill(6) + ".bmp"
            # count_samples[indx] = count_samples[indx] + 1
            # try:
            #     import cv2
            #     cutout = cutout.astype('uint8')
            #     cutout = cv2.resize(cutout, (64, 64), interpolation=cv2.INTER_CUBIC)
            #     # good_image += 1

            path='cutout_'+ str(in_data['boxlabel'][boxes]["x1"])+in_data['boxlabel'][boxes]["class"]
            self.saveToPgm(cutout,path)
            # except Exception as e :
            #     print(e)
            #     # buggy_image += 1
            #     print("buggy image")

"""
if __name__ == '__main__':
    # hashkey = '89e1a7231d563b99c44ae8f9d6a1d2ac'
    # uri = 'mongodb://read:read@luas100x.lu.de.conti.de/'
    uri='mongodb://ozd0127u:1234'
    # db='vodca_labels'
    db='Labels'
    # col='snapshot_20180326'
    col='Shape_Latest_7_Classes'

    ImageInstance=Image()
    ImageInstance.getConnection(db,col,uri)
    ImageInstance.getImage()

