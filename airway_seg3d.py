import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
import scipy.io as io


def load_dicom(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    vol = sitk.GetArrayFromImage(image)
    vol = np.transpose(vol, axes=(1, 2, 0))
    return vol


def segment_airway(dicom_dir):
    plt.figure()
    # Load CT volume
    volume = load_dicom(dicom_dir)
    slice_to_visualize = volume.shape[2] // 2
    plt.subplot(1, 4, 1)
    plt.imshow(volume[:, :, slice_to_visualize], cmap='gray')
    plt.title("Input")

    # Air segmentation by thresholding
    intensity_threshold = -250
    air = np.zeros(volume.shape)
    air[volume <= intensity_threshold] = 1
    plt.subplot(1, 4, 2)
    plt.imshow(air[:, :, slice_to_visualize], cmap='gray')
    plt.title("Air")

    # Get outer air from the region-label at [50,50,z//2]
    regions = ndimage.label(air)[0]
    outer_air_label = regions[50, 50, slice_to_visualize]
    outer_air = np.zeros_like(air)
    outer_air[regions == outer_air_label] = 1
    plt.subplot(1, 4, 3)
    plt.imshow(outer_air[:, :, slice_to_visualize], cmap='gray')
    plt.title("Outer Air")

    # Get inner airway as: air - outer_air
    airway = air - outer_air
    plt.subplot(1, 4, 4)
    plt.imshow(airway[:, :, slice_to_visualize], cmap='gray')
    plt.title("Airway")
    plt.waitforbuttonpress()

    return airway

dicom_dir = "test_dicoms/11413295"
airway = segment_airway(dicom_dir)

# save <airway> using scipy.io.savemat (faster) or npy.save (slower)
io.savemat('output.mat', {'airway': airway})


