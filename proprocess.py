import logging
from scipy import misc
import params
# Preprocess
def preprocess():
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
#        try:
#            img = misc.imread(img_dir)
#        except IOError:
#            logging.warning('Image at '+img_dir+' can not be read')
#            continue
        cur_info['img_dir'] = img_dir
    
    word_dict[seg[0]] = cur_info
    return word_dict

if __name__ == "__main__":
    word_dict = preprocess()