## Flixstock Visual Search Assignment
        

### Steps -             

- Generate index of feature vectors for all the images in the search database               
``` 
python3 generate_index_vectors.py --path ../visualsimilarity/bottoms_resized_png/ --model models/resnet50.pth
```

- Query Image to get 10 similar images      
```
python3 get_similar_images.py --input_image ../visualsimilarity/bottoms_resized_png/35468915NIR.png --model models/resnet50.pth
```     