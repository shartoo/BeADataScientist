import cv2

index = 0

def filter_per_group(first_image,layer,blur_size,ratio_k,sigma):
    global index
    '''
        对一组图像内的图像进行平滑
    :param first_image:     一组图像内的第一张图像
    :param layer:            一组图像内有多少层
    :param blur_size:        高斯平滑半径
    :param ratio_k:           比例
    :param sigma:             高斯平滑参数sigma
    :return:                 一组被高斯平滑之后的图像
    '''
    imgs = [first_image]
    for l in range(1,layer):
        sigma = ratio_k**(l-1)*sigma #平滑系数 k^{l-1}*σ
        print("sigama is:\t",sigma)
        dst = cv2.GaussianBlur(first_image, (blur_size, blur_size), sigma, blur_size)
        cv2.imwrite("group%d_layer%d.jpg"%(index,l),dst)
        imgs.append(dst)
    index = index + 1
    return imgs

def image_pyramid(image_file,group,layer,blur_size,ratio_k,sigma=1.6):
    '''
            SIFT中高斯金字塔
    :param image_file:              需要执行高斯金字塔的原图 文件路径
    :param group:                   图像金字塔分组数
    :param layer:                   图像金字塔每组层数目
    :param blur_size:               高斯平滑的尺寸
    :param ratio_k:                 高斯平滑系数
    :param sigma:                   高斯平滑因子
    :return:                        完整的高斯金字塔
    '''
    image =cv2.imread(image_file)
    shape0,shape1,c = image.shape
    print("image shape is:\t",c,shape0,shape1)
    group_images = []

    for g in range(group):
        if len(group_images)==0:
            # 原图扩大一倍，第一组 第一层
            first_image = cv2.pyrUp(image, dstsize=(2 * shape0, 2 * shape1))
        else:
            # 上一组 倒数第三层图像,降采样
            first_image = group_images[-1][-3]
            shape0, shape1, c = first_image.shape
            print("new shape:\t",int(shape0/2), int(shape1//2))
            first_image = cv2.pyrDown(first_image,dstsize=(int(shape0/2), int(shape1//2)))
            print("group is:\t",g)
            cv2.imwrite("pramid_group%d.jpg"%(g),first_image)
        one_group_imgs = filter_per_group(first_image, layer, blur_size, ratio_k, sigma)
        group_images.append(one_group_imgs)
    return group_images

def show(img):

    print("number of image pramid is:\t",len(img))
    for i in range(len(img)):
        for j in range(len(img[i])):
            cv2.imshow("pramid_group%d_layer%d"%(i,j),img[i][j])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
group =4
layer = 3
blur_size =3
ratio_k = 2
imgs = image_pyramid("lnea.jpg",group,layer,blur_size,ratio_k)
show(imgs)

