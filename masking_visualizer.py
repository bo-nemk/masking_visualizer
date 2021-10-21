import pyperceptualaudio as ppa
import numpy as np
import matplotlib.pyplot as ppl

target_fs = 8000 
N_segment = int(0.2 * target_fs)
print(N_segment)

par_spectrum = ppa.spectrum_calibration(1, 100)
par_calibration_setup = ppa.calibration_setup(target_fs, 1000, N_segment, par_spectrum)
par_model = ppa.par_model(target_fs, N_segment, ppa.erbspace(0, target_fs / 2, 64), 4)
par_model.calibrate(par_calibration_setup, par_spectrum)

x = par_calibration_setup.masker_sinusoid_freq
q = np.zeros(x.shape)

N_transform = N_segment / 2 + 1

# Generate detectabilities
dx = np.diag(par_model.matrix(x))
dq = np.diag(par_model.matrix(q))

# Generate masking thresholds
tx = par_spectrum.to_spl_representation(1 / (dx * N_transform))
tq = par_spectrum.to_spl_representation(1 / (dq * N_transform))

# Axis
fr = np.fft.rfftfreq(N_segment, d = 1 / target_fs)
print(fr.size)
print(x.size)

ppl.semilogx(fr, tq)

target_fs = 8000 / 2
N_segment = int(0.2 * target_fs)
print(N_segment)

par_spectrum = ppa.spectrum_calibration(1, 100)
par_calibration_setup = ppa.calibration_setup(target_fs, 1000, N_segment, par_spectrum)
par_model = ppa.par_model(target_fs, N_segment, ppa.erbspace(0, target_fs / 2, 64), 4)
par_model.calibrate(par_calibration_setup, par_spectrum)

x = par_calibration_setup.masker_sinusoid_freq
q = np.zeros(x.shape)

N_transform = N_segment / 2 + 1

# Generate detectabilities
dx = np.diag(par_model.matrix(x))
dq = np.diag(par_model.matrix(q))

# Generate masking thresholds
tx = par_spectrum.to_spl_representation(1 / (dx * N_transform))
tq = par_spectrum.to_spl_representation(1 / (dq * N_transform))

# Axis
fr = np.fft.rfftfreq(N_segment, d = 1 / target_fs)
print(fr.size)
print(x.size)

ppl.semilogx(fr, tq)
ppl.show()
# import pandas as pd
# pd.DataFrame.from_dict(
#         {
#             "fr" : fr, 
#             "tq" : tq,
#             "tx" : tx,
#             "dq" : dq,
#             "dx" : dx,
#             # Don't ask me why... It's a weird one
#             "x"  : par_spectrum.to_spl_representation((1 / (N_transform)) * x)
#         }
#     ).to_csv("masking.csv", index=False)
