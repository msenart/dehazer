def get_pixel_intensity(pixel):
    '''
    returns the intensity of one pixel. The list must be of length 3 and values must be computed in this order R-G-B.
    '''
    if (len(pixel)!=3):
        if (pixel[0] < 256 and pixel[1] < 256 and pixel[2] < 256):
            return Exception(f"pixel : {pixel} - is not a RGB pixel !")
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
