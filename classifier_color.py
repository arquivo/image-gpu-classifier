from classifier_base import ClassifierBase

from colorthief import MMCQ

class ClassifierColor(ClassifierBase):


    def __init__(self):
        super().__init__()
        self.do_process_image = False

    def rgb_to_hex(self, rgb):
        return "#{0:02x}{1:02x}{2:02x}".format(rgb[0], rgb[1], rgb[2])
 
    def classify(self, image_datas):
        if image_datas == []:
            return []
        output = []
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
                        if not (r > 250 and g > 250 and b > 250):
                            valid_pixels.append((r, g, b))
                cmap = MMCQ.quantize(valid_pixels, color_count)
                dominant_color = cmap.palette[0]
                output.append({"dominant_color": self.rgb_to_hex(dominant_color)})
            except:
                output.append({"dominant_color": "#c6c6c6"})
        return output
