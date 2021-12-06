from PIL import Image
import sys
import numpy as np
import json
import cv2
from sklearn.cluster import MiniBatchKMeans
from colorthief import ColorThief
from pixelator import pixelator
import operator
from collections import defaultdict
import re


def color_quantization(src, dest, clusters=2):
    image = cv2.imread(src)
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clf = MiniBatchKMeans(n_clusters=clusters)
    labels = clf.fit_predict(image)
    quantized = clf.cluster_centers_.astype("uint8")[labels]

    #print(clf.cluster_centers_.astype("uint8"))

    """
    dominant_colors = []
    for center in clf.cluster_centers_.astype("uint8"):
        red_scaled = center[0]
        green_scaled = center[1]
        blue_scaled = center[2]
        
        dominant_colors.append((
            red_scaled * red_std / 255,
            green_scaled * green_std / 255,
            blue_scaled * blue_std / 255
        ))

    plt.imshow([dominant_colors])
    plt.show()
    """

    quantized = quantized.reshape((h,w,3))
    image = image.reshape((h,w,3))
    quantized = cv2.cvtColor(quantized, cv2.COLOR_LAB2BGR, cv2.CV_8U)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR, cv2.CV_8U)

    #cv2.imwrite(dest, np.hstack([image, quantized]))
    #cv2.imwrite(dest, quantized)

    color_thief = ColorThief(src)
    palette = color_thief.get_palette(color_count=2)
    print('HERE')
    print(palette)
    return palette

def kmeans_color_quantization(src, dest, clusters=2, rounds=1):
    # Open the source image
    image = cv2.imread(src)

    h, w = image.shape[:2]
    samples = np.zeros([h * w, 3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              rounds,
                                              cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res = res.reshape((image.shape))


    #cv2.imshow('result', result)
    #cv2.waitKey()

    # Save the pixelized image
    im = Image.fromarray(res)
    im.save(dest)

def pixelator(src, dest, palette):
    """
    palette = [
        (45, 50, 50),  # black
        (240, 68, 64),  # red
        (211, 223, 223),  # white
        (160, 161, 67),  # green
        (233, 129, 76),  # orange
    ]


    sensitivity_multiplier = 10

    size = (32, 32)

    output = pixelator(src, palette, size)

    output.resize_out_img().save_out_img(path=dest, overwrite=True)
    """
    while len(palette) < 256:
        palette.append((0, 0, 0))

    flat_palette = reduce(lambda a, b: a + b, palette)
    assert len(flat_palette) == 768

def pixelate_img(src, dest):

    # Open the source image
    img = Image.open(src)

    # Quantize the image to reduce the number of colors in the palette
    #img = img.quantize(4)  # remove
    #img = img.convert('P', palette=Image.ADAPTIVE, colors=4) # delete later

    # Downsample the image using PIL resizing filter
    #
    # Possible filters are: PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.ANTIALIAS
    #
    # Antialias seems to preserve depth better
    # Nearest seems to have more contrast
    # Bilinear and Bicubic seem to be "flatter"
    img_downsampled = img.resize((32, 32), resample=Image.ANTIALIAS)

    # Scale the downsampled image back up to original size
    #result = img_downsampled.resize(img.size, Image.NEAREST)
    result = img_downsampled.resize((64, 64), Image.NEAREST) # change back

    # Convert the result back into RGB color space
    result = result.convert('RGB')

    # Save the pixelized image
    result.save(dest)

def getClosestColor(pixel, rgb_palette): # Get the closest color for the pixel
    closest_color = None
    cost_init = 1000000
    pixel = np.array(pixel)
    for color in rgb_palette:
        color = np.array(color)
        cost = np.sum((color - pixel)**2)
        if cost < cost_init:
            cost_init = cost
            closest_color = color
    return closest_color

def apply_palette(img_path, dest, palette):
    """
    color_set = ['#1D1D21', '#B02E26', '#5E7C16', '#835432', '#3C44AA', '#8932B8', '#169C9C', '#9D9D97', '#474F52',
                 '#F38BAA',
                 '#80C71F', '#FED83D', '#3AB3DA', '#C74EBD', '#F9801D', '#F9FFFE']  # Given Colorset
    color_set_rgb = [ImageColor.getrgb(color) for color in color_set]  # RGB Colorset
    """

    im = cv2.imread(img_path)  # read input image

    height, width, channels = im.shape
    im_out = np.zeros((height, width, channels))

    color_set = set()
    rgba_arr = []

    for y in range(0, height):
        cur_row = []
        for x in range(0, width):
            closest_color = getClosestColor(im[y, x], palette)
            im_out[y, x, :] = closest_color

            print(closest_color)
            col = (closest_color[0], closest_color[1], closest_color[2])

            color_set.add(col)

            cur_row.append([closest_color[0], closest_color[1], closest_color[2], 255])

        rgba_arr.append(cur_row)

    # Saving the array in a text file
    file = open('C:\\Users\\willc\\Documents\\Knitting\\599-Knitting-Complete\\colorwork_sample_images\\output_images\\lotus.txt', "w")
    content = str(rgba_arr)
    file.write(content)
    file.close()

    #return im_out
    #print(im_out)
    #im = Image.fromarray(im_out)
    #im.save(dest)
    cv2.imwrite(dest, im_out)

    print("Unique colors in apply_palette output")
    print(len(color_set))

def get_color_array_2(img_path, color_array_dest_path='C:\\Users\\willc\\Documents\\Knitting\\599-Knitting-Complete\\colorwork_sample_images\\output_images\\lotus.txt'):
    im = cv2.imread(img_path)  # read input image

    height, width, channels = im.shape
    im_out = np.zeros((height, width, channels))

    rgba_arr = []
    color_set = set()

    for y in range(0, height):
        cur_row = []
        for x in range(0, width):
            color_tuple = im[y,x]

            cur_row.append([color_tuple[0], color_tuple[1], color_tuple[2]])
            color_set.add((color_tuple[0], color_tuple[1], color_tuple[2]))
            #color_set.add(color_tuple)

        rgba_arr.append(cur_row)

    # Saving the array in a text file
    file = open(color_array_dest_path, "w")
    content = str(rgba_arr)
    file.write(content)
    file.close()

    print('Set length')
    print(len(color_set))

    ###
    """
    imgobj = Image.open(img_path)
    pixels = imgobj.convert('RGBA')
    rgba_arr = []
    #print(np.asarray(pixels).shape)

    color_set = set()

    for i in range(imgobj.height):
        cur_row = []
        for j in range(imgobj.width):
            coord = (j,i)
            r,g,b = imgobj[j][i]
            a=255
            #r, g, b, a = pixels.getpixel(coord)
            cur_row.append([r, g, b, a])

            color_set.add((r,g,b,a))

        rgba_arr.append(cur_row)

    # Saving the array in a text file
    file = open(color_array_dest_path, "w+")
    content = str(rgba_arr)
    file.write(content)
    file.close()

    print('Set length')
    print(len(color_set))
    """


def get_color_array(img_path, color_array_dest_path='C:\\Users\\willc\\Documents\\Knitting\\599-Knitting-Complete\\colorwork_sample_images\\output_images\\lotus.txt'):
    imgobj = Image.open(img_path)
    pixels = imgobj.convert('RGBA')
    rgba_arr = []
    #print(np.asarray(pixels).shape)

    color_set = set()

    for i in range(imgobj.height):
        cur_row = []
        for j in range(imgobj.width):
            coord = (j,i)
            #r,g,b = imgobj[j][i]
            #a=255
            r, g, b, a = pixels.getpixel(coord)
            cur_row.append([r, g, b, a])

            color_set.add((r,g,b,a))

        rgba_arr.append(cur_row)

    # Saving the array in a text file
    file = open(color_array_dest_path, "w+")
    content = str(rgba_arr)
    file.write(content)
    file.close()

    print('Set length')
    print(len(color_set))
    #for elem in color_set:
    #    print(elem)

    #a_set = set(rgba_arr)
    #number_of_unique_values = len(a_set)
    #print(number_of_unique_values)



    """
    pixels = imgobj.convert('RGBA')
    data = imgobj.getdata()
    arr = np.array(imgobj)
    print(arr.shape)

    rows = arr.shape[1]
    cols = arr.shape[0]

    rgba_arr = []

    for i in range(rows):
        cur_row = []
        for j in range(cols):
            pixel = arr[j][i]
            cur_row.append([pixel[0], pixel[1], pixel[2], 255])

        rgba_arr.append(cur_row)

    print(rgba_arr)
    #np.savetxt('lotus.txt', rgba_arr)
    #a_file = open("test.txt", "w")
    #for row in an_array:
    #    np.savetxt(a_file, row)

    #a_file.close()

    #lists = rgba_arr.tolist()
    #json_str = json.dumps(rgba_arr)

    #numpyData = {"array": rgba_arr}
    #encodedNumpyData = json.dumps(numpyData)

    #with open(color_array_dest_path, 'w') as f:
    #    json.dump(encodedNumpyData, f, ensure_ascii=False)

    #print()
    #np.savetxt(color_array_dest_path, rgba_arr, delimiter=',')
    file = open(color_array_dest_path, "w+")

    # Saving the array in a text file
    content = str(rgba_arr)
    file.write(content)
    file.close()
    """


if __name__ == "__main__":

    # Collect command line arguments
    path = sys.argv[1]
    dest = sys.argv[2]

    palette=color_quantization(path, dest, 4)  # put back

    # Apply pixelation
    pixelate_img(path, dest)
    #pixelator(path, dest, palette)

    #palette = color_quantization(path, dest, 4)

    #apply_palette(dest, dest, palette)
    apply_palette(dest, "C:\\Users\\willc\\Documents\\Knitting\\599-Knitting-Complete\\colorwork_sample_images\\output_images\\lotus_pic_2.jpg", palette)  # put back

    #color_quantization(dest, dest)

    #kmeans_color_quantization(dest, dest)

    # Get color array
    #get_color_array(dest)
    #get_color_array_2("C:\\Users\\willc\\Documents\\Knitting\\599-Knitting-Complete\\colorwork_sample_images\\output_images\\lotus_pic_2.jpg")
