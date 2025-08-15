import os
from glob import glob

def get_csv(data_root_dir = '/Users/Amanda/Desktop/ePillID-benchmark/mydata', val_fold=3, test_fold=4):
    csv_files =  glob(data_root_dir + '/folds/**/*.csv', recursive=True)
    all_imgs_csv = [x for x in csv_files if x.endswith("all.csv")][0]
    folds = sorted([x for x in csv_files if not x.endswith("all.csv")])
    return all_imgs_csv, folds

def load_data(all_imgs_csv, folds, folds_sorted=True)
    if not folds_sortedÑ
        folds=sorted(folds)
    all_imgs_df = pd.read_csv(all_imgs_csv)
    test_df = pd.read_csv(folds[test_fold])
    val_df = pd.read_csv(folds[val_fold])

    img_dir = 'classification_data'
    for df in [all_images_df, val_df, test_df]:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(data_root_dir, img_dir, x))

    val_test_image_paths = list(val_df['image_path'].values) + list(test_df['image_path'].values)
    train_df = all_imgs_df[~all_imgs_df.isin(val_test_image_paths)].reset_index()
    
    return {'train': train_df, 'val': val_df, 'test': test_df, 'all': all_imgs_df}


def save_model(model_name, curr_epoch, save_dir = '/Users/Amanda/Desktop/PillRecognition/model'):
    os.makedirs(os.path.join(save_dir, model_name, 'embedding'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, model_name, 'classifier'), exist_ok=True)

    torch.save(models['embedding'], os.path.join(save_dir, model_name, 'embedding', 'em_epoch_{:d}.pt'.format(curr_epoch)))
    torch.save(models['classifier'], os.path.join(save_dir, model_name, 'classifier', 'cl_epoch_{:d}.pt'.format(curr_epoch)))
