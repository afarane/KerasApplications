# ref https://deeplearningsandbox.com/how-to-build-an-image-recognition-system-using-keras-and-tensorflow-for-a-1000-everyday-object-559856e04699

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


model = ResNet50(weights='imagenet')

img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))


def predict(model,img,top_n=2):	
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	
	return decode_predictions(preds,top=top_n)[0]
	
	
def plot_preds(image, preds):  

  #image
  plt.imshow(image)
  plt.axis('off')
  
  #bar graph
  plt.figure()  
  order = list(reversed(range(len(preds))))  
  bar_preds = [pr[2] for pr in preds]
  labels = (pr[1] for pr in preds)
  plt.barh(order, bar_preds, alpha=0.5)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0, 1.01)
  plt.tight_layout()
  plt.show()

  
print('-----------------------------------------------------')
preds = predict(model,img)
print(preds)
plot_preds(img,preds) 
print('-----------------------------------------------------')


