from PIL import Image
import sys


def pixelate_img(src, dest):

    # Open the source image
    img = Image.open(src)

    # Quantize the image to reduce the number of colors in the palette
    img = img.quantize(10)

    # Downsample the image using PIL resizing filter
    #
    # Possible filters are: PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.ANTIALIAS
    #
    # Antialias seems to preserve depth better
    # Nearest seems to have more contrast
    # Bilinear and Bicubic seem to be "flatter"
    img_downsampled = img.resize((32, 32), resample=Image.ANTIALIAS)

    # Scale the downsampled image back up to original size
    result = img_downsampled.resize(img.size, Image.NEAREST)

    # Convert the result back into RGB color space
    result = result.convert('RGB')

    # Save the pixelized image
    result.save(dest)


if __name__ == "__main__":

    # Collect command line arguments
    path = sys.argv[1]
    dest = sys.argv[2]

    # Apply pixelation
    pixelate_img(path, dest)
