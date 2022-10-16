from ROOT import TFile
import numpy as np

def load_data(file_path : str) -> dict:
    """Loads relevant polarization data from the .root file at file_path

    Args:
        file_path (str): path where .root file is located

    Returns:
        dict: Dictionary containing numpy arrays of all necessary observables
    """
    
    file = TFile(file_path)
    ntuple = file.Get("ntuple")
    
    EVENT = []
    QDC_EIO0 = []
    TTTL_OP_Beam = []
    TDC_PHOTO_DIODE = []
    
    for n, entry in enumerate(ntuple):
        EVENT.append(n)
        
        QDC_EIO0.append(entry.QDC_EIO0)
        
        TTTL_OP_Beam.append(entry.TTTL_OP_Beam)
        
        # Track TDC data. Note that TDC data is given as a vector which does not always contain a number
        # hence we need to check that there is at least one entry in each TDC vector before trying to grab
        # it otherwise the code will not work.
        # If a TDC vector does not contain an element add np.nan to the array instead this allows for the use 
        # of array operations (np.nan does not).
        if entry.TDC_PHOTO_DIODE_LE_Count > 0:
            TDC_PHOTO_DIODE.append(entry.TDC_PHOTO_DIODE_LE[0])
        else:
            TDC_PHOTO_DIODE.append(np.nan)
        
    # Define conversion factor to properly convert TDC data.
    # According to OnlineHistograms.cpp in rootana src code this value
    # comes from 'Equipment/VME/Settings/V1192_ns_conversion' and seems
    # to be a hardware specific value.
    v1192_to_ns = 0.098765625
    
    # Make dictionary containing numpy arrays of all observables also take care of scaling
    dict = {
        "EVENT"             : np.asarray(EVENT),
        "TTTL_OP_Beam"      : np.asarray(TTTL_OP_Beam) / 1e6,               # us
        "TDC_PHOTO_DIODE"   : np.asarray(TDC_PHOTO_DIODE) * v1192_to_ns,    # ns
        "QDC_EIO0"          : np.asarray(QDC_EIO0)
    }
    
    return dict