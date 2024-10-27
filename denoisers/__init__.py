import importlib
import os
import denoisers

# Automatically import any Python files in the denoisers/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('denoisers.' + module)

def get_denoiser(args):
    model = args.model
    if model=='N2N' or model=='Neighbor2Neighbor':
        return denoisers.denoise_N2N.denoiser()
    elif model=='N2S' or model=='Noise2Self':
        return denoisers.denoise_N2S.denoiser()
    elif model=='UDVD' or model=='UDVD_mf' or model=='UDVD_multiframe':
        return denoisers.denoise_UDVD.denoiser()
    elif model=='UDVD_sf' or model=='UDVD_singleframe':
        return denoisers.denoise_UDVD_singleframe.denoiser()
    else:
        raise(KeyError('Model not recognized'))
