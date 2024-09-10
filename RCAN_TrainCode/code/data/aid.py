import os

from data import common
from data import srdata

import numpy as np
import torch
import torch.utils.data as data

class AID(srdata.SRData):
    def _init_(self, args, train=True):
        super(AID, self)._init_(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        # Parcours des sous-dossiers dans HR
        for root, _, files in os.walk(self.dir_hr):
            for file in files:
                # Ajouter l'image HR à la liste
                if file.endswith(self.ext):  # Par exemple, .png
                    list_hr.append(os.path.join(root, file))

                # Parcours des images LR (similaire à HR)
                for si, s in enumerate(self.scale):
                    lr_file = os.path.join(self.dir_lr, f'X{s}', root.split(os.sep)[-1],
                                           file)  # Assurez-vous que le chemin correspond à LR
                    if os.path.exists(lr_file):
                        list_lr[si].append(lr_file)

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        # Adapter les répertoires de AID
        self.apath = os.path.join(dir_data, '/mydataset')  # Assurez-vous que "AID" est le dossier de base de vos données
        self.dir_hr = os.path.join(self.apath, 'HR')  # Dossier contenant les images HR
        self.dir_lr = os.path.join(self.apath, 'LR')  # Dossier contenant les images LR
        self.ext = '.jpg'  # Extension de vos fichiers (ajustez si ce n'est pas .png)

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def _len_(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx