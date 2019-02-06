import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import numpy as np
import json
import matplotlib.pyplot as plt

def explore_h5(hdf_file):

    """Traverse all datasets across all groups in HDF5 file."""

    import h5py

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                for h in h5py_dataset_iterator(item, path):
                    yield h

    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)

            groups = ['data','faceId', 'label', 'normal']
            for g in groups:
                if g in path:
                    print np.array(f[g])

    return None



def explore_json(json_file):
    data=[]
    with open(json_file, "w") as f:
        json.dump(data, f)
    return data



def parse_log_file(log_file):
    #EPOCH XXX EVALUATION and EPOCH XXX EVALUATION WHOLE SCENE occur together
    file = open(log_file, 'r')

    # make sure zip(items)[0] are unique!
    items = [('**** EPOCH', ['mean loss', 'accuracy']),
             ('EVALUATION ----', ['eval mean loss', 'eval point accuracy vox', 'eval point avg class acc vox', \
                                  'eval point accuracy', 'eval point avg class acc', 'eval point calibrated average acc']),
             ('EVALUATION WHOLE SCENE----', ['eval whole scene mean loss', 'eval whole scene point accuracy vox', 'eval whole scene point avg class acc vox', \
                                             'eval whole scene point accuracy', 'eval whole scene point avg class acc', 'eval whole scene point calibrated average acc vox'])]
    values = {}
    for item in items:
        values[item[0]] = {}
        for subitem in item[1]:
            values[item[0]][subitem] = []

    parsing = False
    idx_item = None
    for line in file:

        # always be ready to process a new chunk
        for i,item in enumerate(items):
            if item[0] in line:
                noItem = 0
                idx_item = i
                parsing = True
                break

        # parse through the file for however long until new chunk encountered
        if parsing:
            if line.split(':')[0] in items[idx_item][1]:
                values[items[idx_item][0]][line.split(':')[0]].append(float(line.split(':')[-1]))

    #pprint(values)
    # for v,k in values.items():
    #     for a,b in k.items():
    #         print(v,a,len(b))

    fig, axes = plt.subplots(1,3)
    fig.suptitle(log_file.split('/')[-1], size=14)
    for v, ax in zip(values.keys(),axes):
        for a,b in values[v].items():
            ax.plot(b, label=a)

            # annotate values on plot
            for x,y in zip(range(len(b)), b):
                if 'EVALUATION' in v:
                    if x%10==0: ax.text(x,y,str(y)[:4])
                elif 'EPOCH' in v:
                    if x%(15*10)==0: ax.text(x,y,str(y)[:4])

        ax.set_title(v)
        ax.legend()
        ax.set_xlabel('epoch')

        # handle different epoch frequencies
        if 'EVALUATION' in v:
            ax.set_xticks(np.arange(0, len(b), 10))
            ax.set_xticklabels(np.arange(0, len(b), 10)*5)
        elif 'EPOCH' in v:
            ax.set_xticks(np.arange(0, len(b), 15*20)) #bc each epoch has 15 printouts
            ax.set_xticklabels(np.arange(0,len(b), 15*20)/15)



    plt.show()

    return values