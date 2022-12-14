## Flixstock Visual Search Assignment
        

### Steps -    

- Download given dataset - https://drive.google.com/file/d/1OCvfi5L_znC3xGGyH_hXEYEKSGcRleHU/view
- Download model from https://drive.google.com/file/d/1oUuMZTED8E0YJWDvfm630dAJ03hk_25z/view?usp=sharing
and put in models folder


- Generate index of feature vectors for all the images in the search database               
use "--path" for input images folder from which have to query similar images              
use "--model" to provide model path               
``` 
python3 generate_index_vectors.py --path ../visualsimilarity/bottoms_resized_png/ --model models/resnet50.pth
```

- Query Image to get 10 similar images  
Example -           
give your image path in "--input_image" and model path in "--model"

```
python3 get_similar_images.py --input_image ../visualsimilarity/bottoms_resized_png/35468915NIR.png --model models/resnet50.pth
```     

#### Possible Improvements -
- Currently model used for feature extraction is pretrained ResNet50 , performance can be improved by using custom model trained on fashion images. We can use DeepFashion dataset to train our model.
- Code structure can be improved to make it more readable and customizable.
- Speed can be improved by moving all CPU ops to GPU ops.
- Alternate methods can be tested for finding similarity score.

