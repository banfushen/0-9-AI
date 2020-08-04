#coding:utf-8
from PIL import Image, ImageChops
import os

def ImgResize(Img,ScaleFactor):
    ImgSize = Img.size; #获得图像原始尺寸
    NewSize = [int(ImgSize[0]*ScaleFactor),int(ImgSize[1]*ScaleFactor)];  #获得图像新尺寸,保持长宽比
    Img = Img.resize(NewSize);     #利用PIL的函数进行图像resize,类似matlab的imresize函数
    return Img;        

def ImgResizeTo(Img,NewSize):
    Img = Img.resize(NewSize);     #利用PIL的函数进行图像resize,类似matlab的imresize函数
    return Img;      

#旋转
def ImgRotate(img,Degree):
    im2 = img.convert('RGBA')
    #旋转图像
    rot = im2.rotate(Degree)
    #与旋转图像大小相同的白色图像
    fff = Image.new('RGBA',rot.size, (255,255,255,255,))
    #使用alpha层的rot作为掩码创建一个复合图像
    out = Image.composite(rot,fff,rot)
    return out

#利用PIL的函数进行水平以及上下镜像
def ImgLRMirror(Img):
    return Img.transpose(Image.FLIP_LEFT_RIGHT)

def ImgTBMirror(Img):
    return Img.transpose(Image.FLIP_TOP_BOTTOM)

#平移
def ImgOffSet(img,xoff,yoff):
    width, height = img.size
    img = img.convert('RGBA')
    c = ImageChops.offset(img,xoff,yoff)
    c.paste((255,255,255,255),(0,0,xoff,height))
    c.paste((255,255,255,255),(0,0,width,yoff))
    return c

if __name__ == "__main__":
    # Img = Image.open('C:/Users/bfs/Desktop/learning_ai/0/0_0.png')
    # img2 = ImgOffSet(Img, 10, 10)
    # img2.save('C:/Users/bfs/Desktop/learning_ai/out/out.png')
    # input("press enter to exit\n")
    print("Starting...")
    move_img = [
        (0,1),(0,2),(1,1),(1,2),(2,2),
    ]

    rotate_img = [
        -30, -27, -24, -21, -18, -15, -12, -9, -6, -3,
        30, 27, 24, 21, 18, 15, 12, 9, 6, 3,
    ]

    m_dir = 'C:/Users/bfs/Desktop/learning_ai'
    for x in range(0,10):
        img_dir = os.path.join(m_dir, str(x))
        for i in range(10):
            pre_img_name = str(x) + "_"
            img_name = pre_img_name + str(i) + '.png'
            img_path = os.path.join(img_dir, img_name)
            rotate_len = len(rotate_img)
            for move_i in range(len(move_img)):
                for rotate_i in range(rotate_len):
                    file_name = pre_img_name + str(10+100*i + move_i*rotate_len + rotate_i) + ".png"
                    print(file_name, i, move_i*rotate_len, rotate_i)
                    out_dir = m_dir+"/out/"+str(x)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    file_name = os.path.join(out_dir, file_name)
                    Img = Image.open(img_path)
                    img2 = ImgOffSet(Img, move_img[move_i][0], move_img[move_i][1])
                    img2 = ImgRotate(img2, rotate_img[rotate_i])
                    img2.save(file_name)
            pass
    input("press enter to exit\n")