import cv2
import numpy as np
from blob import Blob as blob
from roi import ROI as roi

class Image_Processor:
    def __init__(self):
        self._path = "../data/1280x720/"
        self._images_number = 28
        self._image_origin = 0
        self._image = 0
        self._image_grayscale = 0
        self._image_white_top_hat = 0
        self._image_separeded = 0
        self._blob_list = []
        self._roi_list = []

    # download original image and save it and another image that is top third part of original image
    def source_image_downloader(self, image_name):
        self._image_origin = cv2.imread(image_name)
        self._image = self._image_origin[0:240, 0:1280]

    # convert image to grayscale
    def convert_to_grayscale(self):
        self._image_grayscale = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)

    # using morphology white top-hat operator, which delete not necessary details
    def white_tophat(self, kernel):
        self._image_white_top_hat = cv2.morphologyEx(self._image_grayscale, cv2.MORPH_TOPHAT, kernel)

    # with using some threshold parameter filtering image to separate objects
    def object_separator_tozero(self, thrsh, max):
        ret1, threshold = cv2.threshold(self._image_white_top_hat, thrsh, max, cv2.THRESH_TOZERO)
        self._image_separeded = threshold

    # on separated picture finding blobs and their parameters
    def blob_detector(self):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Filter by Color
        params.filterByColor = True
        params.blobColor = 255

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 64
        params.maxArea = 480

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(self._image_separeded)
        for kp in keypoints:
            radius = kp.size / np.pi ** 0.5
            blob_x = kp.pt[0]
            blob_y = kp.pt[1]
            step = radius / 2 ** 0.5
            # analyse area around the blob centroid to find it average color
            color_area = self._image[int(blob_y - step):int(blob_y + step), int(blob_x - step):int(blob_x + step)]
            median_color_per_row = np.median(color_area, axis=0)
            median_color = np.median(median_color_per_row, axis=0)
            # analyse intensity around the blob to determine in which direction area should be caught
            sum_up = 0
            sum_down = 0
            for i in range(1, 10):
                try:
                    sum_up += self._image_grayscale[int(blob_y) + int(radius) + i][int(blob_x)]
                    sum_down += self._image_grayscale[int(blob_y) - int(radius) - i][int(blob_x)]
                except:
                    pass
            if sum_up > sum_down:
                direct = "up"
            else:
                direct = "down"
            self._blob_list.append(blob(int(blob_x), int(blob_y), radius, direct, median_color))

    # with using of blobs and their parameters finding ROIs (regions of interest on grayscale image)
    def roi_intensity_based(self):
        for blob in self._blob_list:
            step = int(blob.radius * 1.3)
            self._image_grayscale[blob.y - step:blob.y + step, blob.x - step:blob.x + step] = 0
            try:
                # if direction = "up" than catch area under the blob (for green traffic light)
                if blob.direction == "up":
                    self._roi_list.append((roi(
                        self._image_grayscale[blob.y - int(4 * blob.radius):blob.y + int(1.5 * blob.radius),
                        blob.x - int(1.5 * blob.radius):blob.x + int(1.5 * blob.radius)],
                        blob
                    )))
                # if direction = "up" than catch area below the blob (for red traffic light)
                elif blob.direction == "down":
                    self._roi_list.append((roi(
                        self._image_grayscale[blob.y - int(1.5 * blob.radius):blob.y + int(4 * blob.radius),
                        blob.x - int(1.5 * blob.radius):blob.x + int(1.5 * blob.radius)],
                        blob
                    )))
            except:
                pass

    # with using of rois (each of them include blob as parameter) finding traffic light on original image
    def find_traffic_light_intensity_based(self):
        # finding average intensity of ROIs of some picture
        average_intensities = []
        sum_intensity = 0
        count_intensity = 0
        for roi in self._roi_list:
            for j in range(len(roi.image)):
                for k in range(len(roi.image[j])):
                    if roi.image[j][k] != 0:
                        sum_intensity += roi.image[j][k]
                        count_intensity += 1
            try:
                average_intensities.append(int(sum_intensity / count_intensity))
            except:
                pass
            sum_intensity = 0
            count_intensity = 0
        average_trashold = 140
        for i in range(len(average_intensities)):
            if average_intensities[i] < average_trashold:
                # average (median) color values of blob and other parameters of it, which is in blob
                average_blob_blue = self._roi_list[i].blob.median_color[0]
                average_blob_green = self._roi_list[i].blob.median_color[1]
                average_blob_red = self._roi_list[i].blob.median_color[2]
                blob_x = self._roi_list[i].blob.x
                blob_y = self._roi_list[i].blob.y
                direction = self._roi_list[i].blob.direction
                step = 40
                # if green is main color - than traffic light is green
                if average_blob_green > average_blob_red and average_blob_green > average_blob_blue and direction == "up":
                    cv2.rectangle(self._image_origin, (int(blob_x - step), int(blob_y - step * 2)),
                                  (int(blob_x + step), int(blob_y + step / 2)), (0, 255, 0), 2)
                # if red is main color - than traffic light is red
                elif average_blob_red > average_blob_green and average_blob_red > average_blob_blue and direction == "down":
                    cv2.rectangle(self._image_origin, (int(blob_x - step), int(blob_y - step / 2)),
                                  (int(blob_x + step), int(blob_y + step * 2)), (0, 0, 255), 2)

    def images_runner(self):
        for i in range(1, self._images_number+1):
            self.source_image_downloader(self._path + "image%i.png" % i)
            self.convert_to_grayscale()
            self.white_tophat(np.ones((20, 20), np.uint8))
            self.object_separator_tozero(30, 255)
            self.blob_detector()
            self.roi_intensity_based()
            self.find_traffic_light_intensity_based()
            cv2.imshow("im", self._image_origin)
            cv2.waitKey(0)
            self._roi_list = []
            self._blob_list = []

    def video_runner(self):
        cap = cv2.VideoCapture(self._path + 'video.mp4')
        while (cap.isOpened()):
            ret, image = cap.read()
            self._image_origin = image
            self._image = self._image_origin[0:240, 0:1280]
            self.convert_to_grayscale()
            self.white_tophat(np.ones((20, 20), np.uint8))
            self.object_separator_tozero(30, 255)
            self.blob_detector()
            self.roi_intensity_based()
            self.find_traffic_light_intensity_based()
            cv2.imshow('frame', self._image_origin)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self._roi_list = []
            self._blob_list = []
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_processor = Image_Processor()
    # to run on images use
    image_processor.video_runner()
    # image_processor.video_runner()
