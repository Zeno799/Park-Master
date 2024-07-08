from PIL import Image

image_path = 'img2.png'
image = Image.open(image_path)
grayscale_image = image.convert('L')

grayscale_image.save('/Users/niteshregmi/Desktop/untitled folder/grey/grayscale_image.jpg')

grayscale_image.show()


# this is the dependency [pip install Pillow] bitch