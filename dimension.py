import nibabel as nib

path = ''

atlas = nib.load(path)

atlas_dimension = atlas.get_fdata()
print(atlas_dimension.shape)

#atlas_dimension = atlas.agg_data()

#atlas


