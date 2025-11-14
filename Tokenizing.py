import json 
import string
from collections import defaultdict
'''
Default dicionary :
It returns a dictionary-like object that automatically provides a default value 
for missing keys, based on the specified callable, instead of raising a KeyError.
'''

class Parse_MSCOCO:
  def __init__(self, annotation_file, image_directory):
    with open(annotation_file,'r') as file: # reading json file
      coco = json.load(file)
    
    #Dictionaries to store image and annotation data 
    self.image_dictionary = {} # connects image ID to corresponding image information, getting images based on their ID 
    #self.annotations_dictionary = defaultdict(list) # stores image id with list of annotations
    #self.annotations_ID = {} # connects annotation ID to the corresponding annotation information 
    self.caption_dictionary = defaultdict(list) # stores image id with list of captions 

    #Extract values and store them into the dictionaries with their corresponding IDs
    for ann in coco['annotations']:
      #self.annotations_dictionary[ann['image_id']].append(ann)
      #self.annotations_ID[ann['id']] = ann
      self.caption_dictionary[ann['image_id']].append(ann['caption'])

    for img in coco['images']:
      self.image_dictionary[img['id']] = img
  
  #Function to get a list of all image ids 
  def get_imgIds(self):
      return list(self.image_dictionary.keys())

  #Function to get all captions dictionary
  def get_captions(self):
    return self.caption_dictionary

  # Fuction to get list of images with corresponding captions 
  def get_image_with_caption(self, image_ids):
    image_ids = image_ids if isinstance(image_ids,list) else [image_ids]# If image_ids is not a list, single id value
    return [(img_id,self.caption_dictionary[img_id]) for img_id in image_ids] # return tuple list of selected images with their captions 
  
  #Function to formate the captions of specific image(s) with start and end to tonekize later on 

  def get_formatted_captions(self, image_ids):
    image_ids = image_ids if isinstance(image_ids, list) else [image_ids]  # check if single ID or list of tthem

    form_captions = {}

    # Get the corresponding images and their captions using get_image_with_caption
    image_captions = self.get_image_with_caption(image_ids)
    
    for img_id, captions in image_captions: # loop through the dictionary 
        form_captions[img_id] = [] #define empty list image caption pair variable 
        for caption in captions:
            # getting rid of punctuation and convert to lowercase
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            caption = caption.replace("-", " ")
            caption = caption.lower().split()  # Split into words
            caption = ['<start>'] + caption + ['<end>']
            caption = " ".join(caption)
            form_captions[img_id].append(caption)

    return form_captions # return images with corresponding split list captions
    
    # Get img file names 
  def get_img_file_names(self, image_ids):
      image_ids = image_ids if isinstance(image_ids,list) else [image_ids] # If image_ids is not a list, single id value
      return [self.image_dictionary[img_id]['file_name'] for img_id in image_ids]
  
import pickle
import tensorflow as tf
from collections import defaultdict

# Get paths
img_train_path = "./train2014/train2024"
ann_train_path = "./annotations_trainval2014/annotations/captions_train2014.json"

ann_val_path = "./annotations_trainval2014/annotations/captions_val2014.json"
img_val_path = "./val2014"

# Parse data
parser_train = Parse_MSCOCO(ann_train_path, img_train_path)
parser_val = Parse_MSCOCO(ann_val_path, img_val_path)

# Tokenize captions
def tokenize_captions(parser):
    image_ids = parser.get_imgIds()
    formatted_captions = parser.get_formatted_captions(image_ids)
    file_names = parser.get_img_file_names(image_ids)

    # Storing all captions into a single list
    all_captions = []
    for caption_list in formatted_captions.values(): # loop through values
        all_captions.extend(caption_list)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<oov>",filters='') #For those words which are not found
    tokenizer.fit_on_texts(all_captions)

    # Tokenize all individual captions using the real file names to match image feature extraction
    tokenized_captions = {}

    for img_id, file_name in zip(formatted_captions.keys(), file_names):
        # Remove ".jpg" from file_name
        file_name = file_name.split('.')[0]
        tokenized_captions[file_name] = tokenizer.texts_to_sequences(formatted_captions[img_id])

    return tokenizer, tokenized_captions

tokenizer, tokenized_captions = tokenize_captions(parser_train)
tokenizer_val, val_tokenized_captions = tokenize_captions(parser_val)

tok_train_path = "./tokenized_train_captions2.pkl"
with open(tok_train_path, "wb") as f:
    pickle.dump(tokenized_captions, f)

tok_path = "./tokenizer2.pkl"
with open(tok_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved.")
vocabulary_size = len(tokenizer.word_index)+1

tok_val_path = "./tokenized_val_captions2.pkl"
with open(tok_val_path, "wb") as f:
    pickle.dump(val_tokenized_captions, f)