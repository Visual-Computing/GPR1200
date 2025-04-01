import numpy as np
import os
from utils import * 


class GPR1200:

    """GPR1200 class
    
    The dataset contains 12k images from 1200 diverse categories. 
    """
    
    _base_dir = None
    
    _image_data = None
    _ground_truth = None
    
    _iterator_index = 0
    
    def __init__(self, base_dir):
        """
        Load the image information from the drive
        
        Parameters
        ----------
        base_dir : string 
            GPR1200 base directory path
        """
        self._base_dir = base_dir
        
        gpr10x1200_cats, gpr10x1200_files = [], []

        data = sorted(os.listdir(base_dir), key=lambda a: int(os.path.basename(a).split("_")[0]))
        for file in data:
            file_path = os.path.join(base_dir, file)
            cat = os.path.basename(file).split("_")[0]
            gpr10x1200_cats.append(cat)
            gpr10x1200_files.append(file_path)

        gpr10x1200_cats, gpr10x1200_files = np.array(gpr10x1200_cats), np.array(gpr10x1200_files)
        
        #sorted_indx = np.argsort(ur10x1000_files)
        self._image_files = gpr10x1200_files#[sorted_indx]
        self._image_categories = gpr10x1200_cats#[sorted_indx]

    @staticmethod
    def __name__():
        """
        Name of the  dataset
        """
        return "GPR1200"
        
    def __str__(self): 
        """
        Readable string representation
        """
        return "" + self.__name__() + "(" + str(self.__len__()) + ") in " + self.base_dir
    
    def __len__(self): 
        """
        Amount of elements
        """
        return len(self._image_data)

    @property
    def base_dir(self):
        """
        Path to the base directory
        
        Returns
        -------
        path : str
            Path to the base directory
        """
        return self._base_dir

    @property
    def image_dir(self):
        """
        Path to the image directory
        
        Returns
        -------
        path : str
            Path to the image directory
        """
        return self._base_dir + "images/"

    @property
    def image_files(self): 
        """
        List of image files. The order of the list is important for other methods.
        
        Returns
        -------
        file_list : list(str)
            List of file names
        """
        return self._image_files


    def evaluate(self, features=None, indices=None, compute_partial=False, float_n=4, metric="cosine"):
        """
        Compute the mean average precision of each part of this combined data set. 
        Providing just the 'features' will assume the manhatten distance between all images will be computed 
        before calculating the mean average precision. This metric can 
        be changed with any scikit learn 'distance_metric'. 
      
         
        Parameters
        ----------
        features : ndarray 
            matrix representing the embeddings of all the images in the dataset
        indices: array-lile, shape = [n_samples_Q, n_samples_DB]
            Nearest neighbours indices 

        """
        
        cats = self._image_categories

        if (indices is None) & (features is None):
            raise ValueError("Either indices or features_DB has to be provided ")

        if indices is None:
            aps = compute_mean_average_precision(cats, features_DB=features, metric=metric)
        if features is None:
            aps = compute_mean_average_precision(cats, indices=indices, metric=metric)

        all_map = np.round(np.mean(aps), decimals=float_n)

        if compute_partial: 

            cl_map = np.round(np.mean(aps[:2000]), decimals=float_n)
            iNat_map = np.round(np.mean(aps[2000:4000]), decimals=float_n)
            sketch_map = np.round(np.mean(aps[4000:6000]), decimals=float_n)
            instre_map = np.round(np.mean(aps[6000:8000]), decimals=float_n)
            sop_map = np.round(np.mean(aps[8000:10000]), decimals=float_n)
            faces_map = np.round(np.mean(aps[10000:]), decimals=float_n)
            
            return all_map, cl_map, iNat_map, sketch_map, instre_map, sop_map, faces_map

        return all_map