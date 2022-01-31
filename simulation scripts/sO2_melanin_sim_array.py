from ippai.simulate.simulation import simulate
from ippai.simulate.structures import create_epidermis_layer
from ippai.simulate import SegmentationClasses
from ippai.utils import OpticalTissueProperties
from ippai.utils import TissueSettingsGenerator
from ippai.utils import CHROMOPHORE_LIBRARY
from ippai.utils import SPECTRAL_LIBRARY
from ippai.utils import Chromophore
from ippai.utils import Tags
from ippai.io_handling import load_hdf5
import numpy as np
import time
import sys

print("[starting ID] "+str(sys.argv[1]))
SAVE_PATH = "/SET_PATH"
MCX_BINARY_PATH = "/SET_PATH_TO/mcx"
EXP_STR = "sO2_melanin_skin"

VOLUME_WIDTH_IN_MM = 32
VOLUME_HEIGHT_IN_MM = 11
SPACING = 0.15
ILLUMINATION_OFFSET_Y_IN_MM = 10
snellangle_gel_pad = np.arcsin(np.sin(np.deg2rad(45)))

WAVELENGTHS = np.arange(680, 981, 20)

VOLUMES = [int(sys.argv[1])]

ILLUMINATION_POSITIONS = []
illum_x_pos_MM = np.arange(-12,13,8)
for i in illum_x_pos_MM:
    ILLUMINATION_POSITIONS.append([
        int(VOLUME_WIDTH_IN_MM/SPACING / 2.0 + i/SPACING),
        int(VOLUME_WIDTH_IN_MM/SPACING / 2.0 
            + ILLUMINATION_OFFSET_Y_IN_MM/SPACING), 
        0])

def create_background_optical_properties(rCu, sulfate_volume_fraction):
    """
    Water background with a scattering parameter that can be tuned using 
    b_mie and f_ray using the power law formula from Jacques et al. 
    as used in mcxyz.
    """
    background_settings_generator = TissueSettingsGenerator()
    background_settings_generator.append(
        key="scatterer", 
        value=Chromophore(spectrum=SPECTRAL_LIBRARY.WATER,
            volume_fraction=1.0, 
            musp500=OpticalTissueProperties.MUSP500_BACKGROUND_TISSUE*10,
            b_mie=OpticalTissueProperties.BMIE_BACKGROUND_TISSUE, 
            f_ray=OpticalTissueProperties.FRAY_BACKGROUND_TISSUE, 
            anisotropy=0.9))
    background_settings_generator.append(
        key="OxyHemoglobin", 
        value=Chromophore(spectrum=SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
                          volume_fraction = rCu*sulfate_volume_fraction, 
                          musp500=0.0, b_mie=0.0, f_ray=0.0, anisotropy=0.9))
    background_settings_generator.append(
        key="DeOxyHemoglobin", 
        value=Chromophore(spectrum=SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN,
                          volume_fraction = (1-rCu)*sulfate_volume_fraction, 
                          musp500=0.0, b_mie=0.0, f_ray=0.0, anisotropy=0.9))
    return background_settings_generator.get_settings()


def create_target_optical_properties(rCu):
    # Create an instance of a TissueSettingsGenerator
    tissue_settings_generator = TissueSettingsGenerator()
    # Appending chromophores
    
    tissue_settings_generator.append(
        key="scatterer",
        value=Chromophore(spectrum=SPECTRAL_LIBRARY.WATER,
            volume_fraction=1.0,                        
            musp500=OpticalTissueProperties.MUSP500_BACKGROUND_TISSUE*10,
            b_mie=OpticalTissueProperties.BMIE_BACKGROUND_TISSUE,
            f_ray=OpticalTissueProperties.FRAY_BACKGROUND_TISSUE,
            anisotropy=0.9))
    tissue_settings_generator.append(
        key="OxyHemoglobin", 
        value=Chromophore(spectrum=SPECTRAL_LIBRARY.OXYHEMOGLOBIN,
                          volume_fraction=rCu,
                          musp500=0.0,
                          b_mie=0.0,
                          f_ray=0.0,
                          anisotropy=0.9))
    tissue_settings_generator.append(
        key="DeOxyHemoglobin", 
        value=Chromophore(spectrum=SPECTRAL_LIBRARY.DEOXYHEMOGLOBIN,
                          volume_fraction=(1-rCu), 
                          musp500=0.0,
                          b_mie=0.0,
                          f_ray=0.0,
                          anisotropy=0.9))
    return tissue_settings_generator.get_settings()

def create_background(rCu, sulfate_volume_fraction):
    rnd_bg_dict = dict()
    rnd_bg_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_BACKGROUND
    rnd_bg_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = create_background_optical_properties(
        rCu, sulfate_volume_fraction)
    rnd_bg_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.WATER
    return rnd_bg_dict

def create_tube(rCu=None, x_min=None, x_max=None, z_min=None, z_max=None, r=0.25):
    tube_dict = dict()
    tube_dict[Tags.STRUCTURE_TYPE] = Tags.STRUCTURE_TUBE
    tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MIN_MM] = z_min
    tube_dict[Tags.STRUCTURE_CENTER_DEPTH_MAX_MM] = z_max
    tube_dict[Tags.STRUCTURE_RADIUS_MIN_MM] = r
    tube_dict[Tags.STRUCTURE_RADIUS_MAX_MM] = r
    tube_dict[Tags.STRUCTURE_TUBE_CENTER_X_MIN_MM] = x_min
    tube_dict[Tags.STRUCTURE_TUBE_CENTER_X_MAX_MM] = x_max
    tube_dict[Tags.STRUCTURE_TISSUE_PROPERTIES] = create_target_optical_properties(rCu=rCu)
    tube_dict[Tags.STRUCTURE_SEGMENTATION_TYPE] = SegmentationClasses.BLOOD
    return tube_dict

def create_lsd_trip_tissue_volume_exp(VOLUME_CENTER_MM, RND_SEED):
    # SMOFlipid 1.5% experimental properties
    # from analyse_smoflipid with rho 20 mm
    #background_scattering = 265.74933
    #background_anisotropy = 0.8 # buggy mean calculation in ippai causes this to mean 0.9 background anisotropy
    #background_b_mie = 1.3154256
    #background_f_ray = 6.88908e-17

    TUBE_DEPTH_min_MM = 0.5
    TUBE_DEPTH_max_MM = 10.5
    TUBE_MAX_OFFSET = 12
    TUBE_INNER_RADIUS_MM = 0.4
    
    np.random.seed(RND_SEED)
    number_of_vessels_1 = int(np.random.uniform(3, 10, 1))
    number_of_vessels_2 = int(np.random.uniform(3, 10, 1))
    rCu_tube_1 = float(np.random.uniform(0.0,1.0,1)[0])
    print("=======sO2_tube_1=", rCu_tube_1)
    rCu_tube_2 = float(np.random.uniform(0.0,1.0,1)[0])
    print("=======sO2_tube_2=", rCu_tube_2)
    rCu_bg = float(np.random.uniform(0.0,1.0,1)[0])
    print("=======sO2_bg=", rCu_bg)
    sulfate_volume_fraction = float(np.random.uniform(0.01,0.05,1)[0])
    print("=======bvf_bg=", sulfate_volume_fraction)

    tissue_dict = dict()
    tissue_dict["background"] = create_background(
        rCu=rCu_bg, 
        sulfate_volume_fraction=sulfate_volume_fraction
    )

    tissue_dict["skin"] = create_epidermis_layer()
    for i in range(number_of_vessels_1):
        tissue_dict["tube_1_"+str(i)] = create_tube(
            rCu=rCu_tube_1, 
            x_min=VOLUME_CENTER_MM - TUBE_MAX_OFFSET,
            x_max=VOLUME_CENTER_MM + TUBE_MAX_OFFSET,
            z_min=TUBE_DEPTH_min_MM,
            z_max=TUBE_DEPTH_max_MM,
            r=TUBE_INNER_RADIUS_MM)
    for i in range(number_of_vessels_2):
        tissue_dict["tube_2_"+str(i)] = create_tube(
            rCu=rCu_tube_2,
            x_min=VOLUME_CENTER_MM - TUBE_MAX_OFFSET,
            x_max=VOLUME_CENTER_MM + TUBE_MAX_OFFSET,
            z_min=TUBE_DEPTH_min_MM,
            z_max=TUBE_DEPTH_max_MM,
            r=TUBE_INNER_RADIUS_MM)
    return tissue_dict

for volume_ in VOLUMES:
    for ill_idx in range(len(ILLUMINATION_POSITIONS)):
        settings = {
            # Set the general propeties of the simulated volume.
            Tags.RANDOM_SEED: volume_,
            Tags.VOLUME_NAME: "MI-LSD/MI-LSD_"+EXP_STR+"_" + str(volume_) + "_ill_" + str(ill_idx),
            Tags.SIMULATION_PATH: SAVE_PATH,
            Tags.SPACING_MM: SPACING,
            Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_IN_MM,
            Tags.DIM_VOLUME_Y_MM: VOLUME_WIDTH_IN_MM,
            Tags.AIR_LAYER_HEIGHT_MM: 0,
            Tags.GELPAD_LAYER_HEIGHT_MM: 10,

            # Set the optical forward model.
            Tags.RUN_OPTICAL_MODEL: True,
            Tags.WAVELENGTHS: WAVELENGTHS,
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 2e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: MCX_BINARY_PATH,
            Tags.OPTICAL_MODEL: Tags.MODEL_MCX,
            Tags.ILLUMINATION_DIRECTION: [0,
                                          -np.sin(snellangle_gel_pad),
                                          np.cos(snellangle_gel_pad)],
            Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_GAUSSIAN,
            # from mcxlab/README.txt:
            # 'gaussian' [*] - a collimated gaussian beam, 
            # srcparam1(1) specifies the waist radius (in voxels)
            Tags.ILLUMINATION_PARAM1: [40, 0, 0, 0],
            Tags.ILLUMINATION_POSITION: ILLUMINATION_POSITIONS[ill_idx],
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 10,

            # The following parameters tell the script that we do not want any extra
            # modelling steps
            Tags.RUN_ACOUSTIC_MODEL: False,
            Tags.APPLY_NOISE_MODEL: False,
            Tags.PERFORM_IMAGE_RECONSTRUCTION: False,
            Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW: True,

            # Add the structures to be simulated to the tissue
            Tags.STRUCTURES: create_lsd_trip_tissue_volume_exp(
                VOLUME_CENTER_MM = VOLUME_WIDTH_IN_MM/2, 
                RND_SEED = volume_)}
        files = simulate(settings)
