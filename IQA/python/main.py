import sys
import os
import csv
import time
from brisque import BRISQUE as BR

def get_file_realpath(src, *tar):
    '''
    返回图片文件的路径
    Parameters
    ----------
    src: sring. 图片root目录
    tar: 图片类型文件后缀. [".jpg",“.png”]
    '''
    for root, _, files in os.walk(src):
        for fn in files:
            fn_name, fn_ext = os.path.splitext(fn)
            if fn_ext.lower() not in tar:
                continue
            
            yield os.path.join(root, fn)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("Please give input argument of the image path.")
        print("Arguments expected: <image_path>")
        print("--------------------------------")
        print("Exiting")
        sys.exit(0)

    f = open('quality_result.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(['path', 'quality_score'])
    imageNames = get_file_realpath(sys.argv[1], *[".jpg", ".png", ".bmp", ".jpeg"])
    T = BR()
    start_time1 = time.time()
    for img_path in imageNames:
        q_score = T.test_measure_BRISQUE(img_path)
        csv_write.writerow([img_path.split("/")[-2]  +'/'+ img_path.split("/")[-1], q_score])
    print("Toal Time cost is: {}".format(str(time.time() - start_time1)))
    f.close()
