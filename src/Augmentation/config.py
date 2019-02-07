
from paths import *
from collections import defaultdict

objects_hash = defaultdict(list)
objects_hash['bed'] = '1.0-0.7333333333333333-0.47058823529411764'
objects_hash['chair'] = '0.4-0.4-0.4'

hashed_color_names = defaultdict(list)
hashed_color_names['1.0-0.7333333333333333-0.47058823529411764'] = 'bed'
hashed_color_names['0.4-0.4-0.4'] = 'chair'

new_objects_address = defaultdict(list)
new_objects_address['bed'] = [bed_pc_name]
