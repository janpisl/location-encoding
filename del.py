import pandas as pd
import geopandas as gpd





def sample_rows(df, column, n_rows):
    """
    Return a DataFrame which contains n_rows for each unique value in 'column', 
    randomly sampled. If there are less rows than n_rows for some value in 'column', 
    return all rows that have that value.

    Arguments:
        df: pandas.DataFrame
        column: str
        n_rows: int
        
    Returns:
        pandas.DataFrame
    """
    return df.groupby(column).apply(lambda x: x.sample(min(len(x), n_rows))).reset_index(drop=True)

import pdb


annotation = pd.read_csv("/home/jan/Documents/EPFL/project_2/data/labels/dataset_v5/labels_grid_5_split_no_masolele.csv")
annotation['dst_size'] = None

total_size = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
total_size.sort(reverse=True)
train_dataset = annotation.loc[annotation.split.isin([0,1,2,3])].reset_index(drop=True)

for size in total_size:
    #pdb.set_trace()
    examples_per_class = int(size/len(train_dataset.driver.unique()))
    annotation_for_size = sample_rows(train_dataset, 'driver', examples_per_class)
    if len(annotation_for_size) < size:
        random_samples = train_dataset.loc[~train_dataset.sampleid.isin(annotation_for_size.sampleid)].sample((size - len(annotation_for_size)))
        annotation_for_size = pd.concat([annotation_for_size, random_samples])
    
    
    annotation.loc[annotation.sampleid.isin(annotation_for_size.sampleid), 'dst_size'] = size
    train_dataset = annotation_for_size


pdb.set_trace()
print()
