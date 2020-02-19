import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import glob
class detector1:
    def __init__(self):
        self.cfg='cfg/yolov3.cfg'
        self.data='cfg/coco.data'
        self.weights='weights/yolov3.weights'
        self.source='test/'
        self.output='output'
        self.conf_thres=0.6
        self.nms_thres=0.5
        self.fourcc='mp4v'
        self.half='store_true'
        self.device=''
        self.view_img='store_true'
        self.img_size = (416,416)
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)
        self.model = Darknet(self.cfg, self.img_size).cuda()
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model,self.weights)
        self.model.to(self.device).eval()
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
            torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=True)
            return
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()
        print("done!!!")

    def detect(self,im0s):
        centers=[]
        box_detects=[]
        obj_types=[]
        out, source, weights, half, view_img = self.output, self.source, self.weights, self.half, self.view_img
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')  
        save_img=True      
        #dataset = LoadImages(source, img_size=self.img_size, half=self.half)
        classes = load_classes(parse_data_cfg(self.data)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(3)]
        t0 = time.time()
        # for path, img, im0s, vid_cap in dataset:
        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0
        t = time.time()
        img = torch.from_numpy(img).cuda()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        if self.half:
            pred = pred.float()
        pred = non_max_suppression(pred,self.conf_thres, self.nms_thres)
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *x, conf, _, cls in det:
                   if(classes[int(cls)]=="car" or classes[int(cls)]=="motor") :
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    obj_types.append(classes[int(cls)])
                    top=c1[1]
                    left=c1[0]
                    right=c2[0]
                    bottom=c2[1]
                    box_detects.append(np.array([left, top, right, bottom]))

                    x_center = (left + right) / 2
                    y_center = (top + bottom) / 2

                    centroid = np.array([[x_center], [y_center]])
                    centers.append(np.round(centroid))

                 
                    # cv2.imshow("image",im0[c1[1]:c2[1],c1[0]:c2[0]]) get liciense
                    #cv2.imwrite("data_char/"+str(d)+".jpg",im0[c1[1]:c2[1],c1[0]:c2[0]])

                    # cv2.waitKey(0)
                    # if True:  # Write to file
                    #     with open('a' + '.txt', 'w+') as file:
                    #         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                    # save_img=True
                    # if save_img or view_img:  # Add bbox to image
                    #     label = '%s %.2f' % (classes[int(cls)], conf)
                    #     plot_one_box(x, im0, label=label, color=colors[0])
        return im0s,centers, box_detects, obj_types

 
        
