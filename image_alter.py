from tkinter import *
from PIL import Image,ImageTk
from images_from_phone import *
import images_from_phone


img = (Image.open("images_from_phone/siffror.png"))

w_r, h_r = img.size
w, h = w_r-20, h_r-40
img2 = img.crop([10, 20, w, h])

def binarize(image_to_transform: Image, threshold):
    # now, lets convert that image to a single greyscale image using convert()
    output_image=image_to_transform.convert("L")
    # the threshold value is usually provided as a number between 0 and 255, which
    # is the number of bits in a byte.
    # the algorithm for the binarization is pretty simple, go through every pixel in the
    # image and, if it's greater than the threshold, turn it all the way up (255), and
    # if it's lower than the threshold, turn it all the way down (0).
    # so lets write this in code. First, we need to iterate over all of the pixels in the
    # image we want to work with
    for x in range(output_image.width):
        for y in range(output_image.height):
            # for the given pixel at w,h, lets check its value against the threshold
            if output_image.getpixel((x,y))< threshold: #note that the first parameter is actually a tuple object
                # lets set this to zero
                output_image.putpixel( (x,y), 0 )
            else:
                # otherwise lets set this to 255
                output_image.putpixel( (x,y), 255 )
    #now we just return the new image
    return output_image
 
def get_color_map(image):
    px = image.load()
    map = [[px[x, y] for x in range(image.width)] for y in range(image.height)]
    
    return map

def change_aspect_ratio(image: Image, width, height):
    w, h = image.width, image.height
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return image

#small_one = change_aspect_ratio(binarize(images_1_5[0], 200), 28, 28)


if __name__ == "__main__":
    root = Tk()
    root.geometry("400x400")

    canvas = Canvas(root)
    canvas.pack(expand=True, fill=BOTH)
    """
    black_and_white = [ImageTk.PhotoImage(binarize(image, 100)) for image in images_1_5]
    for i in black_and_white:
        pass
        #label = Label(canvas, image = i).pack(side=LEFT)
    """


    print(232423)
    root.mainloop()