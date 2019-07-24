import numpy as np
import cv2
import os

train_path = 'data\\train'
val_path = 'data\\val'
test_path = 'data\\test'

row_size = 256
column_size = 256


def preprocessing (datum, shape) :
    datum = cv2.resize(datum, shape, cv2.INTER_AREA)
    datum = datum.astype(np.float32)
    
    return datum


def canny_data(datum) :
    datum = cv2.Canny(datum, 50, 200)

    return datum


def make_data (path) :
    data_list = os.listdir(path)

    for datum in data_list :
        path_tmp = os.path.join(path, datum)
        
        img_tmp = cv2.imread(path_tmp, cv2.IMREAD_COLOR)
        canny_img = canny_data(img_tmp)

        img_tmp = preprocessing(img_tmp, (row_size, column_size))
        canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2RGB)

        canny_img = preprocessing(canny_img, (row_size, column_size))

        img = cv2.hconcat([img_tmp, canny_img])

        os.remove(path_tmp)
        cv2.imwrite(path_tmp, img)

    
def main() :
    make_data(train_path)
    make_data(val_path)
    make_data(test_path)


if __name__ == '__main__' :
    main()