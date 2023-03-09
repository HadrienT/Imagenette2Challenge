import os
import PIL.Image as Image
import argparse

def main() -> None:
  parser = argparse.ArgumentParser(description='Build paths files')
  parser.add_argument('--transformed',type=bool, action=argparse.BooleanOptionalAction,help='Make paths for transformed or raw images')
  
  args = parser.parse_args()
  if args.transformed:
    
    path = 'E:\\imagenette2\\transformed'
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    
    with open(path + '/train.txt', 'w') as f:
        filename = 'train'
        for idx,im_class in enumerate(os.listdir(os.path.join(path,filename))):
            for image in os.listdir(os.path.join(path,filename,im_class)):
              full_path = os.path.join(path,filename,im_class,image) 
              if Image.open(full_path).mode == 'RGB':
                full_path = full_path + ',' + labels[idx] +'\n'
                full_path = full_path.replace('\\','\\\\')
                f.write(full_path)
    f.close()

    with open(path + '/val.txt', 'w') as f:
        filename = 'val'
        for idx,im_class in enumerate(os.listdir(os.path.join(path,filename))):
            for image in os.listdir(os.path.join(path,filename,im_class)):
              full_path = os.path.join(path,filename,im_class,image) 
              if Image.open(full_path).mode == 'RGB':
                full_path = full_path + ',' + labels[idx] +'\n'
                full_path = full_path.replace('\\','\\\\')
                f.write(full_path)
                
    f.close()
    
  else:
  
    path = 'E:\\ML\\Datasets\\imagenette2'
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

    with open(path + '/train.txt', 'w') as f:
        filename = 'train'
        for idx,im_class in enumerate(os.listdir(os.path.join(path,filename))):
            for image in os.listdir(os.path.join(path,filename,im_class)):
              full_path = os.path.join(path,filename,im_class,image) 
              if Image.open(full_path).mode == 'RGB':
                full_path = full_path + ',' + labels[idx] +'\n'
                full_path = full_path.replace('\\','\\\\')
                f.write(full_path)
                # print(full_path)
    f.close()

    with open(path + '/val.txt', 'w') as f:
        filename = 'val'
        for idx,im_class in enumerate(os.listdir(os.path.join(path,filename))):
            for image in os.listdir(os.path.join(path,filename,im_class)):
              full_path = os.path.join(path,filename,im_class,image) 
              if Image.open(full_path).mode == 'RGB':
                full_path = full_path + ',' + labels[idx] +'\n'
                full_path = full_path.replace('\\','\\\\')
                f.write(full_path)
                
    f.close()
if __name__ == '__main__':
  main()