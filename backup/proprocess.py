import logging
from scipy import misc
import params
import numpy as np
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
# Preprocess
def read():
    '''
    Read the images and their infomation, return a dictionary where key is the word id, value is the dictionary
        about that word's information including the image path
    '''
    
    data_dir = params.data_dir
    f = open(data_dir+'words.txt',"r")
    word_list = f.readlines()
    word_dict = {}
    for word in word_list:
        cur_info = {}
        seg = word.split()
        
        id_list = seg[0].split('-')
        if len(seg) != 9:
            if seg[-1] == 'Ps' and seg[-2][-1] == 'M':
                seg.pop(-1)
                seg[-1] = 'MPS'
            else:
                logging.warning('Number of items in word information is not 9, info: '+word+';length:'+str(len(seg)))
                continue
        
        try:
            cur_info['doc_id'] = id_list[0]
            cur_info['page_id'] = id_list[1]
            cur_info['line_id'] = id_list[2]
            cur_info['word_id'] = id_list[3]
        except IndexError:
            logging.warning('Unexpected id list detected')
            continue
        
        if seg[1] == 'ok':
            cur_info['segIsGood'] = True
        elif seg[1] == 'err':
            cur_info['segIsGood'] = False
        else:
            logging.warning('Unexpected segmentation results detected')
            continue
            
        cur_info['graylevel'] = seg[2]
        cur_info['bounding_box'] = [seg[3],seg[4],seg[5],seg[6]]
        cur_info['tag'] = seg[7]
        cur_info['word'] = seg[8]
        
        img_dir = data_dir + id_list[0]+'/'+'-'.join(id_list[:2])+'/'+seg[0]+'.png'
        try:
            img = misc.imread(img_dir)
        except IOError:
            logging.warning('Image at '+img_dir+' can not be read')
            continue
        cur_info['img_dir'] = img_dir
        cur_info['img'] = img   
        word_dict[seg[0]] = cur_info
    return word_dict

def toMat(word_dict):
    max_width,max_height = 0,0
    for id,info in word_dict.items():
        bounding_box = info['bounding_box']
        max_width = max(max_width,int(bounding_box[2]))
        max_height = max(max_height,int(bounding_box[3]))

#    X = np.empty((0,max_height,max_width))
#    X = np.zeros((len(word_dict),max_height,max_width))
    max_width, max_height = 250,750
    X = np.empty((0,max_height,max_width))
    Y = []
    word_ids = []
    i = 0
    for id,info in tqdm(word_dict.items()):
        bounding_box = info['bounding_box']
        if bounding_box[2] > max_width or bounding_box[3] > max_height:
            continue
        img_dir = info['img_dir']
        try:
            img = misc.imread(img_dir)
        except IOError:
            logging.warning('Image at '+img_dir+' can not be read')
            continue
        res = np.zeros((max_height,max_width))
#        X[i] = utils.toMiddle(img,res)
        i += 1
        X = np.concatenate([X,np.expand_dims(utils.toMiddle(img,res),axis=0)])
        Y.append(info['word'])
        word_ids.append(id)
    Y = np.asarray(Y,dtype=int)
    return X,Y,word_ids


if __name__ == "__main__":
    word_dict = read()
#    w,h = [],[]
#    for id,info in word_dict.items():
#        bounding_box = info['bounding_box']
#        w.append(int(bounding_box[2]))
#        h.append(int(bounding_box[3]))
    
    #X,Y,word_ids = toMat(word_dict)