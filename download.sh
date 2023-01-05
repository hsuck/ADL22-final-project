#!/bin/sh
mkdir ./python/models ./python/models/seen ./python/models/unseen
gdown https://drive.google.com/uc\?id\=1DGZ6tP0x4tuWTRH3yFxgXfQvQCN_VlWW -O ./python/models/seen/model.pth
gdown https://drive.google.com/uc\?id\=1zdNqcX6j8AZu2ZzGzTz-vmM3ywGTlFsB -O ./python/models/unseen/model.pth
