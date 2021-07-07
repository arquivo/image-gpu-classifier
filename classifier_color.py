from classifier_base import ClassifierBase

from colorthief import MMCQ

class ClassifierColor(ClassifierBase):
    """Extract dominant color from image in HEX """

    def __init__(self):
        super().__init__()
        self.do_process_image = False

    def rgb_to_hex(self, rgb):
        """Convert RGB representation into an Hex representation"""
        return "#{0:02x}{1:02x}{2:02x}".format(rgb[0], rgb[1], rgb[2])
 
    def classify(self, image_datas):
        """Extract dominant color from mulitple images"""
        if image_datas == []:
            return []
        output = []
        # find top 5 images to work better with images with multiple colors
        color_count = 5
        quality = 10
        for image_data in image_datas:
            try:
                image = image_data.convert('RGBA')
                width, height = image.size
                pixels = image.getdata()
                pixel_count = width * height
                valid_pixels = []
                for i in range(0, pixel_count, quality):
                    r, g, b, a = pixels[i]
                    # If pixel is mostly opaque and not white
                    if a >= 125:
                        # try to avoid very white/bright pixels
                        if not (r > 250 and g > 250 and b > 250):
                            valid_pixels.append((r, g, b))
                cmap = MMCQ.quantize(valid_pixels, color_count)
                dominant_color = cmap.palette[0]
                output.append({"dominant_color": self.rgb_to_hex(dominant_color)})
            except:
                # return a bland gray if the image failed processing
                output.append({"dominant_color": "#c6c6c6"})
        return output
