import struct
import sys
from models import *
from utils.utils import *

size = sys.argv[3]
model_name = sys.argv[2]
model = Darknet('cfg/'+model_name, (size, size))
weights = sys.argv[1]
device = torch_utils.select_device('0')
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)
model = model.eval()

with open(model_name[:-3] + 'wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

