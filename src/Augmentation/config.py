import sys,os
from collections import defaultdict

rel_path = os.path.realpath('')
reconstructed_scene = rel_path + '/Samples/scene0000_01_vh_clean_2.ply'
segmented_reconstructed_scene = rel_path + '/Samples/scene0000_01_vh_clean_2.labels.ply'
bed_pc_name = rel_path + '/Samples/new_bed_clean.ply'

objects_hash = defaultdict(list)
objects_hash['bed'] = '1.0-0.7333333333333333-0.47058823529411764'

new_objects_address = defaultdict(list)
new_objects_address['bed'] = [bed_pc_name]